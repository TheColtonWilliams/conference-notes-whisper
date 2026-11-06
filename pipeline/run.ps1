<#
.SYNOPSIS
    One-shot pipeline runner:
      extract (video -> audio) -> preprocess -> transcribe -> correct -> speakers -> names -> rename -> kb -> index
      -> [review build -> PAUSE for human edits -> review apply] -> archive

    Recordings are collection-based: drop audio under data/recordings/<collection>/
    and the folder name becomes the collection. Completed audio is moved to
    data/processed/<collection>/ by the final archive step so it is never reprocessed.

    After processing, the pipeline pauses for a human-in-the-loop review of
    acronyms/entities (data/glossary.yaml) and speakers (data/speaker-id/people.yaml). Edit the
    files it points you to, then press Enter (or resume later with -Resume).

.EXAMPLE
    .\pipeline\run.ps1                              # full pipeline + review pause
    .\pipeline\run.ps1 -File "keynote.m4a"          # single file (implies -NoReview)
    .\pipeline\run.ps1 -Resume                      # after editing review files
    .\pipeline\run.ps1 -NoReview                    # skip the review checkpoint
#>

param(
    [string]$File = "",
    [switch]$DryRun,
    [switch]$Verbose,
    [switch]$Resume,
    [switch]$NoReview
)

$ErrorActionPreference = "Stop"
$VenvPython = "$PSScriptRoot\..\\.venv\\Scripts\\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found. Create it first with: python -m venv .venv"
    exit 1
}

# Single-file reprocessing is a targeted operation; skip the (global) review pause.
if ($File) { $NoReview = $true }

$FileArg = if ($File) { "`"$File`"" } else { "" }
$DryArg = if ($DryRun) { "--dry-run" } else { "" }
$VerboseArg = if ($Verbose) { "--verbose" } else { "" }

function Run-Step {
    # -AllFiles: the stage works on whole collections (speaker matching, index), so
    # it takes no file argument even in a -File run.
    param([string]$Name, [string]$Module, [string]$ExtraArgs = "", [switch]$AllFiles)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    $StepFileArg = if ($AllFiles) { "" } else { $FileArg }
    $cmd = "& `"$VenvPython`" -m $Module $StepFileArg $DryArg $VerboseArg $ExtraArgs"
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$Name failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

$startTime = Get-Date
$MarkerPath = "$PSScriptRoot\..\data\.review_pending"

if ($Resume) {
    # The human has edited the review files; apply verified data + finish.
    if (-not (Test-Path $MarkerPath)) {
        Write-Error "Nothing to resume: no pending review (data\.review_pending missing). Run without -Resume first."
        exit 1
    }
    Run-Step "Review (apply)" "pipeline.review apply"
}
else {
    Run-Step "Extract Audio" "pipeline.extract"
    Run-Step "Preprocess" "pipeline.preprocess"
    Run-Step "Transcribe" "pipeline.transcribe"
    Run-Step "Correct" "pipeline.correct"
    Run-Step "Speaker ID" "pipeline.speakers" -AllFiles
    Run-Step "Name Proposals" "pipeline.names"
    Run-Step "Rename Speakers" "pipeline.rename_speakers" -AllFiles
    Run-Step "Knowledge Base" "pipeline.kb extract"
    Run-Step "Index" "pipeline.index" -AllFiles

    if (-not $NoReview -and -not $DryRun) {
        Run-Step "Review (build)" "pipeline.review build"

        if ([Environment]::UserInteractive) {
            Write-Host "`n=== REVIEW REQUIRED ===" -ForegroundColor Yellow
            Write-Host "Verify acronyms/entities and speakers before finalizing:" -ForegroundColor Yellow
            Write-Host "  1. data\glossary.yaml      -> set 'status: verified' + fix each 'definition'"
            Write-Host "  2. data\speaker-id\people.yaml -> set 'display_name' (one database across all conferences)"
            Write-Host "     or accept proposed names: python -m pipeline.names --accept <id>  (or --accept-all)"
            Write-Host "  Guide: data\REVIEW.md"
            Write-Host "Press Enter to apply now, or Ctrl-C and later run: .\pipeline\run.ps1 -Resume" -ForegroundColor Yellow
            [void](Read-Host "Press Enter to apply")
            Run-Step "Review (apply)" "pipeline.review apply"
        }
        else {
            Write-Host "`nReview queue built (non-interactive session). Edit data\glossary.yaml + data\speaker-id\people.yaml, then run: .\pipeline\run.ps1 -Resume" -ForegroundColor Yellow
            exit 0
        }
    }
}

Run-Step "Archive" "pipeline.archive"

$elapsed = (Get-Date) - $startTime
Write-Host "`n=== Pipeline complete in $($elapsed.ToString('hh\:mm\:ss')) ===" -ForegroundColor Green
