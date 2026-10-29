"""Runner HD-BET 2.0.1 dengan model dimuat sekali dan jumlah proses terbatas.

Dijalankan dengan Python di .venv-hdbet (bukan diimport ke kernel notebook):
    python src/hdbet_runner.py -i <folder_input> -o <folder_output> --device cuda --tta 1

Isinya mengikuti HD_BET.hd_bet_prediction.hdbet_predict (mode folder, hanya mask, setara CLI
`--save_bet_mask --no_bet_image`). Perbedaannya hanya jumlah proses nnU-Net: CLI memakai
num_processes_preprocessing=4 dan num_processes_segmentation_export=8, yang menghabiskan RAM
di laptop 14 GB. Model, bobot, TTA, dan parameter prediksi sama.
"""
import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="folder berisi *.nii.gz (3D)")
    parser.add_argument("-o", "--output", required=True, help="folder output mask <nama>_bet.nii.gz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tta", type=int, choices=[0, 1], required=True)
    parser.add_argument("--np_preprocessing", type=int, default=1)
    parser.add_argument("--np_export", type=int, default=1)
    parser.add_argument("--sequential", type=int, choices=[0, 1], default=0,
                        help="1 = predict_from_files_sequential (tanpa worker multiprocessing; hemat RAM)")
    parser.add_argument("--deterministic", type=int, choices=[0, 1], default=0,
                        help="1 = cuDNN deterministik (tanpa benchmark) agar mask identik antar-run")
    args = parser.parse_args()
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # harus sebelum CUDA diinisialisasi

    import torch
    from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, nifti_files
    from HD_BET.checkpoint_download import maybe_download_parameters
    from HD_BET.hd_bet_prediction import get_hdbet_predictor

    t0 = time.time()
    maybe_download_parameters()
    predictor = get_hdbet_predictor(use_tta=bool(args.tta), device=torch.device(args.device), verbose=False)
    if args.deterministic:
        # nnUNetPredictor menyalakan cudnn.benchmark=True (pemilihan algoritma bisa beda antar-run -> mask beda
        # beberapa voxel). Model, bobot, dan TTA tidak berubah; hanya cara GPU menghitung.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    t_load = time.time() - t0

    input_files = nifti_files(args.input)
    if not input_files:
        raise SystemExit(f"Tidak ada .nii.gz di {args.input}")
    maybe_mkdir_p(args.output)
    mask_files = [join(args.output, os.path.basename(f)[:-7] + "_bet.nii.gz") for f in input_files]

    t1 = time.time()
    if args.sequential:
        # fungsi nnU-Net yang sama (preprocessing, prediksi, export) dalam satu proses, tanpa worker spawn
        predictor.predict_from_files_sequential(
            [[f] for f in input_files],
            mask_files,
            save_probabilities=False,
            overwrite=True,
            folder_with_segs_from_prev_stage=None,
        )
    else:
        predictor.predict_from_files(
            [[f] for f in input_files],
            mask_files,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=args.np_preprocessing,
            num_processes_segmentation_export=args.np_export,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )
    t_pred = time.time() - t1
    for name in ("dataset.json", "plans.json", "predict_from_raw_data_args.json"):
        p = join(args.output, name)
        if os.path.exists(p):
            os.remove(p)

    missing = [m for m in mask_files if not os.path.exists(m)]
    summary = {"n_inputs": len(input_files), "n_masks": len(mask_files) - len(missing), "missing": missing,
               "model_load_seconds": t_load, "predict_seconds": t_pred, "tta": bool(args.tta),
               "device": args.device, "np_preprocessing": args.np_preprocessing, "np_export": args.np_export,
               "sequential": bool(args.sequential), "deterministic": bool(args.deterministic),
               "cudnn_benchmark": torch.backends.cudnn.benchmark}
    print("HDBET_RUNNER_SUMMARY " + json.dumps(summary))
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
