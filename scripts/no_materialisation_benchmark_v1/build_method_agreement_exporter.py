#!/usr/bin/env python3
"""Build a separately named, untimed MDLovoFit diagnostic exporter."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
MA_ROOT = ROOT / "results/no_materialisation_benchmark_v1/method_agreement_v2"
SOURCE_ROOT = ROOT / "results/no_materialisation_benchmark_v1/04_instrumented_kernel/source"
SOURCE = SOURCE_ROOT / "mdlovofit_timed.f90"
EXPORT_SOURCE = MA_ROOT / "01_mdlovofit_export_validation/source/mdlovofit_export.f90"
EXPORT_BINARY = MA_ROOT / "01_mdlovofit_export_validation/bin/mdlovofit_export"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label} insertion point, found {count}")
    return text.replace(old, new, 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_source() -> None:
    text = SOURCE.read_text()
    text = _replace_once(
        text,
        "             itype, ntype, j, nc, iref\n",
        "             itype, ntype, j, nc, iref, nframes_export, &\n"
        "             nframes_selection, export_selected\n",
        "integer declaration",
    )
    text = _replace_once(
        text,
        "                          lflash(:), best_bije(:), bije(:), resnum(:)\n",
        "                          lflash(:), best_bije(:), bije(:), resnum(:), &\n"
        "                          frame_selection_count(:), ca_to_all(:), final_rank(:)\n",
        "integer allocatable declaration",
    )
    text = _replace_once(
        text,
        "                                   av_best_bije(:), rmsf(:)\n",
        "                                   av_best_bije(:), rmsf(:), internal_evidence(:), &\n"
        "                                   export_rotation(:,:,:), export_translation(:,:), &\n"
        "                                   export_metrics(:,:)\n",
        "real allocatable declaration",
    )
    text = _replace_once(
        text,
        "  character(len=200) :: record, trajfile, rmsffile, atomsfile\n",
        "  character(len=200) :: record, trajfile, rmsffile, atomsfile\n"
        "  character(len=500) :: export_prefix, export_path\n",
        "character declaration",
    )
    text = _replace_once(
        text,
        "  logical :: printperframe, mapfrac, first_bije, print_rmsf, fileexists, endline\n",
        "  logical :: printperframe, mapfrac, first_bije, print_rmsf, fileexists, endline, &\n"
        "             export_enabled\n",
        "logical declaration",
    )
    text = _replace_once(
        text,
        "  trajfile = \"NONE\"\n",
        "  trajfile = \"NONE\"\n"
        "  export_prefix = \"NONE\"\n"
        "  export_enabled = .false.\n",
        "export initialization",
    )
    text = _replace_once(
        text,
        "      case ( \"-perframe\" , \"-pf\" ) \n",
        "      case ( \"-ma-export\" )\n"
        "        call getarg(i+1,record)\n"
        "        export_prefix = trim(adjustl(record))\n"
        "        export_enabled = .true.\n"
        "        i = i + 2\n"
        "      case ( \"-perframe\" , \"-pf\" ) \n",
        "command parser",
    )
    text = _replace_once(
        text,
        '             "-atomsfile" ) \n',
        '             "-atomsfile", &\n'
        '             "-ma-export" ) \n',
        "input-file parser",
    )
    text = _replace_once(
        text,
        "            best_bije(nc), bije(nc), av_best_bije(nc) )\n",
        "            best_bije(nc), bije(nc), av_best_bije(nc), &\n"
        "            frame_selection_count(nc), ca_to_all(nc), final_rank(nc), &\n"
        "            internal_evidence(nc) )\n",
        "diagnostic allocation",
    )
    text = _replace_once(
        text,
        "    av_best_bije(i) = 0.d0\n",
        "    av_best_bije(i) = 0.d0\n"
        "    frame_selection_count(i) = 0\n"
        "    ca_to_all(i) = 0\n"
        "    final_rank(i) = 0\n"
        "    internal_evidence(i) = 0.d0\n",
        "diagnostic initialization",
    )
    text = _replace_once(
        text,
        "        ic = ic + 1\n        xref(ic) = xread\n",
        "        ic = ic + 1\n"
        "        ca_to_all(ic) = iall\n"
        "        xref(ic) = xread\n",
        "CA metadata mapping",
    )
    text = _replace_once(
        text,
        "          ! Summing up the best rmsd found to compute average over frames\n",
        "          ! Count the final best subset once per frame in this untimed export build.\n"
        "          if ( export_enabled .and. .not. mapfrac ) then\n"
        "            do i = 1, n_consider\n"
        "              frame_selection_count(best_bije(i)) = &\n"
        "                  frame_selection_count(best_bije(i)) + 1\n"
        "            end do\n"
        "          end if\n\n"
        "          ! Summing up the best rmsd found to compute average over frames\n",
        "per-frame final-selection counter",
    )
    text = _replace_once(
        text,
        "      call flashsort(av_best_bije, nc, lflash, mflash, bije_new)\n",
        "      if ( export_enabled ) then\n"
        "        nframes_selection = iframe\n"
        "        nframes_export = iframe + 1\n"
        "        internal_evidence = av_best_bije\n"
        "        allocate(export_rotation(3,3,nframes_export), &\n"
        "                 export_translation(3,nframes_export), &\n"
        "                 export_metrics(3,nframes_export))\n"
        "        export_rotation = 0.d0\n"
        "        export_translation = 0.d0\n"
        "        export_metrics = 0.d0\n"
        "      end if\n"
        "      call flashsort(av_best_bije, nc, lflash, mflash, bije_new)\n",
        "pre-sort evidence capture",
    )
    text = _replace_once(
        text,
        "        bije_old(ic) = i\n      end do\n      \n      ! Computing center of mass of reference atoms\n",
        "        bije_old(ic) = i\n"
        "        final_rank(ic) = i\n"
        "      end do\n"
        "      \n      ! Computing center of mass of reference atoms\n",
        "final rank capture",
    )
    text = _replace_once(
        text,
        "            av_rmsd_all = av_rmsd_all + rmsd_all\n            call system_clock(stage_stop)\n",
        "            av_rmsd_all = av_rmsd_all + rmsd_all\n"
        "            if ( export_enabled ) then\n"
        "              export_rotation(:,:,iframe) = u(:,:)\n"
        "              call rotate(xcm,ycm,zcm,u,xtemp,ytemp,ztemp)\n"
        "              export_translation(1,iframe) = xref_cm - xtemp\n"
        "              export_translation(2,iframe) = yref_cm - ytemp\n"
        "              export_translation(3,iframe) = zref_cm - ztemp\n"
        "              export_metrics(1,iframe) = rmsd_low\n"
        "              export_metrics(2,iframe) = rmsd_high\n"
        "              export_metrics(3,iframe) = rmsd_all\n"
        "            end if\n"
        "            call system_clock(stage_stop)\n",
        "transform capture",
    )
    export_block = r'''

  ! Diagnostic exports are intentionally written only after all timing regions stop.
  if ( export_enabled .and. allocated(export_rotation) ) then
    export_path = trim(export_prefix)//"_final_fixed_subset.tsv"
    open(31,file=trim(export_path),status="replace",action="write")
    write(31,"(a)") "atom_index"//achar(9)//"chain"//achar(9)//"residue_id"//achar(9)// &
      "resname"//achar(9)//"final_rank"//achar(9)//"selected_in_final_fixed_subset"// &
      achar(9)//"fraction"//achar(9)//"K"
    do i = 1, nc
      export_selected = 0
      if ( final_rank(i) <= n_consider ) export_selected = 1
      write(31,"(i0,a)",advance="no") i, achar(9)
      write(31,"(a,a)",advance="no") trim(adjustl(pdbstring_left(ca_to_all(i))(22:22))), achar(9)
      write(31,"(a,a)",advance="no") trim(adjustl(pdbstring_left(ca_to_all(i))(23:27))), achar(9)
      write(31,"(a,a)",advance="no") trim(adjustl(pdbstring_left(ca_to_all(i))(18:20))), achar(9)
      write(31,"(i0,a,i0,a,f0.6,a,i0)") final_rank(i), achar(9), export_selected, &
        achar(9), frac, achar(9), n_consider
    end do
    close(31)

    export_path = trim(export_prefix)//"_per_frame_final_selection_frequency.tsv"
    open(32,file=trim(export_path),status="replace",action="write")
    write(32,"(a)") "atom_index"//achar(9)//"chain"//achar(9)//"residue_id"//achar(9)// &
      "resname"//achar(9)//"selected_frame_count"//achar(9)//"eligible_frame_count"// &
      achar(9)//"selection_frequency"
    do i = 1, nc
      write(32,"(i0,a)",advance="no") i, achar(9)
      write(32,"(a,a)",advance="no") trim(adjustl(pdbstring_left(ca_to_all(i))(22:22))), achar(9)
      write(32,"(a,a)",advance="no") trim(adjustl(pdbstring_left(ca_to_all(i))(23:27))), achar(9)
      write(32,"(a,a)",advance="no") trim(adjustl(pdbstring_left(ca_to_all(i))(18:20))), achar(9)
      write(32,"(i0,a,i0,a,es24.16)") frame_selection_count(i), achar(9), nframes_selection, &
        achar(9), dble(frame_selection_count(i))/dble(nframes_selection)
    end do
    close(32)

    export_path = trim(export_prefix)//"_internal_selection_evidence.tsv"
    open(33,file=trim(export_path),status="replace",action="write")
    write(33,"(a)") "atom_index"//achar(9)//"aggregate_improvement_event_count"//achar(9)// &
      "final_rank"//achar(9)//"selected_in_final_fixed_subset"
    do i = 1, nc
      export_selected = 0
      if ( final_rank(i) <= n_consider ) export_selected = 1
      write(33,"(i0,a,es24.16,a,i0,a,i0)") i, achar(9), -internal_evidence(i), &
        achar(9), final_rank(i), achar(9), export_selected
    end do
    close(33)

    export_path = trim(export_prefix)//"_final_transforms.tsv"
    open(34,file=trim(export_path),status="replace",action="write")
    write(34,"(a)") "frame_index"//achar(9)//"rotation_00"//achar(9)//"rotation_01"// &
      achar(9)//"rotation_02"//achar(9)//"rotation_10"//achar(9)//"rotation_11"// &
      achar(9)//"rotation_12"//achar(9)//"rotation_20"//achar(9)//"rotation_21"// &
      achar(9)//"rotation_22"//achar(9)//"translation_x"//achar(9)//"translation_y"// &
      achar(9)//"translation_z"//achar(9)//"least_mobile_RMSD"//achar(9)// &
      "mobile_RMSD"//achar(9)//"all_selection_RMSD"
    do iframe = 1, nframes_export
      write(34,"(i0,15(a,es24.16))") iframe, achar(9), export_rotation(1,1,iframe), &
        achar(9), export_rotation(1,2,iframe), achar(9), export_rotation(1,3,iframe), &
        achar(9), export_rotation(2,1,iframe), achar(9), export_rotation(2,2,iframe), &
        achar(9), export_rotation(2,3,iframe), achar(9), export_rotation(3,1,iframe), &
        achar(9), export_rotation(3,2,iframe), achar(9), export_rotation(3,3,iframe), &
        achar(9), export_translation(1,iframe), achar(9), export_translation(2,iframe), &
        achar(9), export_translation(3,iframe), achar(9), export_metrics(1,iframe), &
        achar(9), export_metrics(2,iframe), achar(9), export_metrics(3,iframe)
    end do
    close(34)
  end if
'''
    text = _replace_once(
        text,
        "  write(error_unit,\"(a,1x,a,1x,es24.16)\") \"NM_TIMING\", \"total_program\", timing_total\n\nend program mdlovofit\n",
        "  write(error_unit,\"(a,1x,a,1x,es24.16)\") \"NM_TIMING\", \"total_program\", timing_total\n"
        + export_block
        + "\nend program mdlovofit\n",
        "post-timing export block",
    )
    EXPORT_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    EXPORT_SOURCE.write_text(text)


def main() -> None:
    generate_source()
    EXPORT_BINARY.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/bin/f95",
        str(SOURCE_ROOT / "common.f90"),
        str(SOURCE_ROOT / "lib.f90"),
        str(EXPORT_SOURCE),
        "-o",
        str(EXPORT_BINARY),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"exporter compilation failed:\n{completed.stdout}\n{completed.stderr}")
    manifest = MA_ROOT / "01_mdlovofit_export_validation/exporter_build_manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                "MDLovoFit untimed method-agreement exporter",
                "compiler: " + subprocess.run(["/usr/bin/f95", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0],
                "source_base: results/no_materialisation_benchmark_v1/04_instrumented_kernel/source/mdlovofit_timed.f90",
                f"source_base_sha256: {_sha256(SOURCE)}",
                f"export_source_sha256: {_sha256(EXPORT_SOURCE)}",
                f"export_binary_sha256: {_sha256(EXPORT_BINARY)}",
                "timing_use: prohibited; exporter is for one untimed diagnostic pass only",
            ]
        )
        + "\n"
    )
    print(EXPORT_BINARY.relative_to(ROOT))


if __name__ == "__main__":
    main()
