"""
QM9 regression: PiNet (pinn library) or ACSF+MLP.
- ACSF: Keras pooling model on Dscribe ACSF features (optional).
"""
import argparse
import os
import shutil
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TARGETS = ["u0_atom", "h298", "homo", "lumo", "gap"]
OUTPUT_SUBDIR = "p_network_outputs"


def _resolve_data_path(filename: str) -> str:
    here = os.path.join(_SCRIPT_DIR, filename)
    if os.path.isfile(here):
        return here
    parent = os.path.join(os.path.dirname(_SCRIPT_DIR), filename)
    if os.path.isfile(parent):
        return parent
    raise FileNotFoundError(
        f"Could not find '{filename}'. Looked in:\n  {here}\n  {parent}\n"
        "Place gdb9.csv and gdb9.sdf next to p_network.py or in the parent folder."
    )


def _pinet_load_target_map(csv_path: str, target_column: str) -> Dict[str, float]:
    csv_df = pd.read_csv(csv_path)
    if "mol_id" not in csv_df.columns:
        raise ValueError("CSV must contain 'mol_id' column.")
    if target_column not in csv_df.columns:
        raise ValueError(f"CSV does not contain target column '{target_column}'.")
    return dict(zip(csv_df["mol_id"].astype(str), csv_df[target_column].astype(np.float32)))


def _pinet_load_molecules(
    sdf_path: str, target_map: Dict[str, float], max_molecules: Optional[int] = None
) -> List[dict]:
    molecules = []
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        mol_id = mol.GetProp("_Name")
        if mol_id not in target_map:
            continue
        conf = mol.GetConformer()
        coords = []
        elems = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])
            elems.append(atom.GetAtomicNum())
        molecules.append(
            {
                "mol_id": mol_id,
                "coord": np.array(coords, dtype=np.float32),
                "elems": np.array(elems, dtype=np.int32),
                "target": np.array([target_map[mol_id]], dtype=np.float32),
            }
        )
        if max_molecules is not None and len(molecules) >= max_molecules:
            break
    if not molecules:
        raise RuntimeError("No molecules loaded for PiNet. Check SDF/CSV.")
    return molecules


def _pinet_make_output_signature():
    return {
        "coord": tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        "elems": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "e_data": tf.TensorSpec(shape=(), dtype=tf.float32),
    }


def _pinet_make_input_fn(
    molecules: List[dict], batch_size: int, shuffle: bool, repeat: bool
):
    from pinn.io import sparse_batch

    output_signature = _pinet_make_output_signature()

    def input_fn():
        def generator():
            for mol in molecules:
                yield {
                    "coord": mol["coord"],
                    "elems": mol["elems"],
                    "e_data": np.float32(mol["target"][0]),
                }

        ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        if shuffle:
            ds = ds.shuffle(min(1000, max(1, len(molecules))), seed=42)
        ds = ds.apply(sparse_batch(batch_size))
        if repeat:
            ds = ds.repeat()
        return ds

    return input_fn


def _pinet_build_model_params(model_dir: str) -> dict:
    return {
        "model_dir": model_dir,
        "network": {
            "name": "PiNet",
            "params": {
                "depth": 4,
                "rc": 4.0,
                "atom_types": [1, 6, 7, 8, 9],
            },
        },
        "model": {"name": "potential_model", "params": {"learning_rate": 1e-3}},
    }


def _pinet_extract_prediction_value(pred: dict) -> float:
    for key in ("predictions", "energy", "y", "output"):
        if key in pred:
            value = pred[key]
            if np.isscalar(value):
                return float(value)
            value = np.asarray(value).reshape(-1)
            if value.size > 0:
                return float(value[0])
    for value in pred.values():
        if np.isscalar(value):
            return float(value)
        arr = np.asarray(value).reshape(-1)
        if arr.size > 0:
            return float(arr[0])
    raise ValueError("Unable to parse PiNet prediction output.")


def run_pinet_one_target(
    target_column: str,
    csv_path: str,
    sdf_path: str,
    out_dir: str,
    train_steps: int,
    batch_size: int,
    test_size: float,
    seed: int,
    max_molecules: Optional[int],
    fresh_start: bool,
    skip_train: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train/eval PiNet; write predictions_<target>.csv (same columns as pinn_qm9)."""
    warnings.filterwarnings("ignore")

    from rdkit import RDLogger
    from pinn import get_model

    RDLogger.DisableLog("rdApp.*")

    print(f"\n=== PiNet target: {target_column} ===")
    target_map = _pinet_load_target_map(csv_path, target_column)
    molecules = _pinet_load_molecules(sdf_path, target_map, max_molecules)
    print("Molecules:", len(molecules))

    rng = np.random.default_rng(seed)
    rng.shuffle(molecules)

    n = len(molecules)
    train_size = int((1.0 - test_size) * n)
    train_size = min(max(1, train_size), n - 1)
    train_molecules = molecules[:train_size]
    test_molecules = molecules[train_size:]
    print("Train:", len(train_molecules), "Test:", len(test_molecules))

    model_dir = os.path.join(out_dir, f"PiNet_{target_column}")
    if fresh_start and os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
        print("Removed:", model_dir)

    params = _pinet_build_model_params(model_dir)
    model = get_model(params)

    if not skip_train:
        train_spec = tf.estimator.TrainSpec(
            input_fn=_pinet_make_input_fn(
                train_molecules, batch_size, shuffle=True, repeat=True
            ),
            max_steps=train_steps,
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=_pinet_make_input_fn(
                test_molecules, batch_size, shuffle=False, repeat=False
            ),
            steps=max(1, len(test_molecules) // batch_size),
        )
        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    else:
        print("Skip train; checkpoint:", model_dir)

    prediction_iter = model.predict(
        input_fn=_pinet_make_input_fn(
            test_molecules, batch_size, shuffle=False, repeat=False
        ),
        predict_keys=["energy"],
    )

    rows = []
    for mol, pred in zip(test_molecules, prediction_iter):
        pred_val = _pinet_extract_prediction_value(pred)
        true_val = float(mol["target"][0])
        rows.append(
            {
                "mol_id": mol["mol_id"],
                "true": true_val,
                "pred": pred_val,
                "abs_error": abs(true_val - pred_val),
            }
        )

    result_df = pd.DataFrame(rows)
    mae = float(result_df["abs_error"].mean())
    rmse = float(
        np.sqrt(np.mean((result_df["true"] - result_df["pred"]) ** 2))
    )
    csv_path_out = os.path.join(out_dir, f"predictions_{target_column}.csv")
    result_df.to_csv(csv_path_out, index=False)
    print("Predictions saved:", csv_path_out)
    print("MAE:", mae, "RMSE:", rmse)
    print(result_df.head(10).to_string(index=False))

    y_true = result_df["true"].to_numpy()
    y_pred = result_df["pred"].to_numpy()
    _save_scatter_single(y_true, y_pred, target_column, out_dir)
    return y_true, y_pred


def extract_geometry(mol):
    conf = mol.GetConformer()
    z_list, r_list = [], []
    for atom in mol.GetAtoms():
        z_list.append(atom.GetAtomicNum())
        pos = conf.GetAtomPosition(atom.GetIdx())
        r_list.append([pos.x, pos.y, pos.z])
    return np.array(z_list, dtype=np.int32), np.array(r_list, dtype=np.float64)


def rdkit_to_ase(mol):
    from ase import Atoms

    z_arr, r_arr = extract_geometry(mol)
    pt = Chem.GetPeriodicTable()
    symbols = [pt.GetElementSymbol(int(z)) for z in z_arr]
    return Atoms(symbols=symbols, positions=r_arr)


def build_acsf_features(suppl, df_indexed):
    from dscribe.descriptors import ACSF

    species = ["H", "C", "N", "O", "F"]
    acsf = ACSF(
        species=species,
        r_cut=6.0,
        g2_params=[[1, 0], [1, 1], [1, 2]],
        g4_params=[[1, 1, 1]],
    )
    max_atoms = max(mol.GetNumAtoms() for mol in suppl if mol is not None)

    x_list = []
    mask_list = []
    mol_ids_ok = []

    for mol in suppl:
        if mol is None:
            continue
        mol_id = mol.GetProp("_Name")
        if mol_id not in df_indexed.index:
            continue
        try:
            atoms = rdkit_to_ase(mol)
            features = acsf.create(atoms)
            n_real = features.shape[0]

            padded = np.zeros((max_atoms, features.shape[1]), dtype=np.float32)
            padded[:n_real, :] = features

            mask = np.zeros((max_atoms,), dtype=np.float32)
            mask[:n_real] = 1.0

            x_list.append(padded)
            mask_list.append(mask)
            mol_ids_ok.append(mol_id)
        except Exception:
            continue

    if not x_list:
        raise RuntimeError("No molecules loaded (ACSF). Check SDF/CSV.")

    return (
        np.asarray(x_list, dtype=np.float32),
        np.asarray(mask_list, dtype=np.float32),
        mol_ids_ok,
        max_atoms,
    )


def atomic_network(n_features: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(n_features,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs)


def build_pooling_model(max_atoms: int, n_features: int) -> tf.keras.Model:
    atomic_model = atomic_network(n_features)
    struct_in = tf.keras.layers.Input(shape=(max_atoms, n_features), name="structure")
    mask_in = tf.keras.layers.Input(shape=(max_atoms,), name="atom_mask")
    atomic_out = tf.keras.layers.TimeDistributed(atomic_model)(struct_in)
    mask_exp = tf.keras.layers.Reshape((max_atoms, 1))(mask_in)
    masked = tf.keras.layers.Multiply()([atomic_out, mask_exp])
    total = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="sum_atomic")(
        masked
    )
    model = tf.keras.Model(inputs=[struct_in, mask_in], outputs=total)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="mse",
        metrics=["mae"],
    )
    return model


def save_scatter_grid(
    parity_by_target: Dict[str, Tuple[np.ndarray, np.ndarray]], out_dir: str
) -> str:
    names = list(parity_by_target.keys())
    n = len(names)
    ncols = min(3, max(1, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False
    )
    for idx, name in enumerate(names):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        yt, yp = parity_by_target[name]
        ax.scatter(yt, yp, alpha=0.3, s=6)
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(name)
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")
    fig.suptitle("Parity plots — test set", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "scatter_grid_all_targets.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_scatter_single(
    y_test_real: np.ndarray, y_pred_real: np.ndarray, target_column: str, out_dir: str
) -> None:
    scatter_path = os.path.join(out_dir, f"scatter_{target_column}.png")
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_real, y_pred_real, alpha=0.35, s=8)
    lims = [
        min(y_test_real.min(), y_pred_real.min()),
        max(y_test_real.max(), y_pred_real.max()),
    ]
    plt.plot(lims, lims, "k--", lw=1, label="ideal")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs True ({target_column})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print("Saved:", scatter_path)


def train_acsf_one_target(
    target_column: str,
    X_data: np.ndarray,
    atom_mask: np.ndarray,
    mol_ids_ok: list,
    y_values: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    max_atoms: int,
    n_features: int,
    out_dir: str,
    epochs: int,
    batch_size: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    tf.keras.backend.clear_session()

    y_mean = float(y_values.mean())
    y_std = float(y_values.std())
    if y_std < 1e-12:
        print(f"Skipping '{target_column}': near-zero variance in labels.")
        return None

    y_norm = (y_values - y_mean) / y_std

    X_train = X_data[idx_train]
    X_test = X_data[idx_test]
    m_train = atom_mask[idx_train]
    m_test = atom_mask[idx_test]
    y_train = y_norm[idx_train]
    y_test = y_norm[idx_test]

    mol_ids_arr = np.array(mol_ids_ok)
    test_mol_ids = mol_ids_arr[idx_test]

    model = build_pooling_model(max_atoms, n_features)
    print(f"\n=== ACSF target: {target_column} ===")
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    model.fit(
        [X_train, m_train],
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    model.evaluate([X_test, m_test], y_test, verbose=1)
    y_pred = model.predict([X_test, m_test], batch_size=128, verbose=0).flatten()

    y_pred_real = y_pred * y_std + y_mean
    y_test_real = y_test * y_std + y_mean
    mae = float(np.mean(np.abs(y_test_real - y_pred_real)))
    rmse = float(np.sqrt(np.mean((y_test_real - y_pred_real) ** 2)))
    print(f"Test MAE ({target_column}):", mae, "RMSE:", rmse)

    weights_path = os.path.join(out_dir, f"weights_acsf_{target_column}.keras")
    model.save_weights(weights_path)
    print("Saved weights:", weights_path)

    pred_df = pd.DataFrame(
        {
            "mol_id": test_mol_ids,
            "true": y_test_real,
            "pred": y_pred_real,
            "abs_error": np.abs(y_test_real - y_pred_real),
        }
    )
    csv_path = os.path.join(out_dir, f"predictions_acsf_{target_column}.csv")
    pred_df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    _save_scatter_single(y_test_real, y_pred_real, f"{target_column}_acsf", out_dir)
    return y_test_real, y_pred_real


def parse_args():
    p = argparse.ArgumentParser(description="QM9: PiNet (default) or ACSF+MLP.")
    p.add_argument(
        "--backend",
        choices=["pinet", "acsf"],
        default="pinet",
        help="pinet: PiNet via pinn (same CSV as pinn_qm9). acsf: Dscribe+Keras.",
    )
    p.add_argument(
        "--targets",
        type=str,
        default=",".join(DEFAULT_TARGETS),
        help=f"Comma-separated targets. Default: {','.join(DEFAULT_TARGETS)}.",
    )
    p.add_argument(
        "--all-csv-columns",
        action="store_true",
        help="Train every numeric column except mol_id (heavy). PiNet only practical with care.",
    )
    p.add_argument("--epochs", type=int, default=35, help="ACSF only.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-steps", type=int, default=600, help="PiNet only.")
    p.add_argument("--max-molecules", type=int, default=None, help="PiNet: cap dataset size.")
    p.add_argument("--fresh-start", action="store_true", help="PiNet: remove checkpoint dir.")
    p.add_argument("--skip-train", action="store_true", help="PiNet: predict from checkpoint.")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = _resolve_data_path("gdb9.csv")
    sdf_path = _resolve_data_path("gdb9.sdf")
    df = pd.read_csv(csv_path)
    df_indexed = df.set_index("mol_id", drop=False)

    if args.all_csv_columns:
        targets = [
            c
            for c in df.columns
            if c != "mol_id" and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not targets:
            raise ValueError("No numeric columns besides mol_id.")
        print("All numeric columns:", targets)
    else:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
        if not targets:
            raise ValueError("No targets.")
        for col in targets:
            if col not in df.columns:
                raise ValueError(f"Unknown column '{col}'.")

    out_dir = os.path.join(_SCRIPT_DIR, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    parity_for_grid: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    if args.backend == "pinet":
        for target_column in targets:
            yt, yp = run_pinet_one_target(
                target_column=target_column,
                csv_path=csv_path,
                sdf_path=sdf_path,
                out_dir=out_dir,
                train_steps=args.train_steps,
                batch_size=args.batch_size,
                test_size=args.test_size,
                seed=args.seed,
                max_molecules=args.max_molecules,
                fresh_start=args.fresh_start,
                skip_train=args.skip_train,
            )
            parity_for_grid[target_column] = (yt, yp)
    else:
        suppl = list(Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True))
        X_data, atom_mask, mol_ids_ok, max_atoms = build_acsf_features(
            suppl, df_indexed
        )
        n_samples = len(mol_ids_ok)
        indices = np.arange(n_samples)
        idx_train, idx_test = train_test_split(
            indices, test_size=args.test_size, random_state=args.seed
        )
        n_features = X_data.shape[2]
        for target_column in targets:
            y_values = df_indexed.loc[mol_ids_ok, target_column].to_numpy(
                dtype=np.float64
            )
            out = train_acsf_one_target(
                target_column=target_column,
                X_data=X_data,
                atom_mask=atom_mask,
                mol_ids_ok=mol_ids_ok,
                y_values=y_values,
                idx_train=idx_train,
                idx_test=idx_test,
                max_atoms=max_atoms,
                n_features=n_features,
                out_dir=out_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            if out is not None:
                parity_for_grid[target_column] = out

    if parity_for_grid:
        grid_path = save_scatter_grid(parity_for_grid, out_dir)
        print("Saved combined figure:", grid_path)


if __name__ == "__main__":
    main()