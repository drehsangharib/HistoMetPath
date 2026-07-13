param(
    [string]$RepositoryRoot = "D:\HistoMetPath\HistoMetPath-repo",
    [int]$NormalCount = 10,
    [int]$TumorCount = 10,
    [switch]$InventoryOnly
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    throw "AWS CLI is required. Install it with: winget install --id Amazon.AWSCLI --exact"
}

$DataRoot = Join-Path $RepositoryRoot "data\camelyon16"
$NormalRoot = Join-Path $DataRoot "training\normal"
$TumorRoot = Join-Path $DataRoot "training\tumor"
$InventoryPath = Join-Path $DataRoot "aws_inventory.txt"
$PlanPath = Join-Path $DataRoot "cohort_download_plan.csv"

New-Item -ItemType Directory -Path $NormalRoot -Force | Out-Null
New-Item -ItemType Directory -Path $TumorRoot -Force | Out-Null

Write-Host "Building public CAMELYON S3 inventory..." -ForegroundColor Cyan
aws s3 ls s3://camelyon-dataset/ --recursive --no-sign-request | Set-Content -LiteralPath $InventoryPath -Encoding UTF8

$InventoryRows = Get-Content -LiteralPath $InventoryPath | ForEach-Object {
    if ($_ -match '^\s*(\S+)\s+(\S+)\s+(\d+)\s+(.+)$') {
        [PSCustomObject]@{
            Date = $Matches[1]
            Time = $Matches[2]
            SizeBytes = [int64]$Matches[3]
            Key = $Matches[4]
        }
    }
}

$NormalCandidates = @(
    $InventoryRows |
        Where-Object { $_.Key -match '(?i)(^|/)CAMELYON16/images/normal_\d+\.tif$' } |
        Sort-Object Key
)

$TumorCandidates = @(
    $InventoryRows |
        Where-Object { $_.Key -match '(?i)(^|/)CAMELYON16/images/tumor_\d+\.tif$' } |
        Sort-Object Key
)

if ($NormalCandidates.Count -lt $NormalCount) {
    throw "Only $($NormalCandidates.Count) normal slide keys were discovered."
}
if ($TumorCandidates.Count -lt $TumorCount) {
    throw "Only $($TumorCandidates.Count) tumor slide keys were discovered."
}

$SelectedNormal = $NormalCandidates | Select-Object -First $NormalCount
$SelectedTumor = $TumorCandidates | Select-Object -First $TumorCount

$Plan = @()
$Plan += $SelectedNormal | ForEach-Object {
    [PSCustomObject]@{
        Class = "normal"
        Key = $_.Key
        SizeBytes = $_.SizeBytes
        Destination = Join-Path $NormalRoot (Split-Path $_.Key -Leaf)
    }
}
$Plan += $SelectedTumor | ForEach-Object {
    [PSCustomObject]@{
        Class = "tumor"
        Key = $_.Key
        SizeBytes = $_.SizeBytes
        Destination = Join-Path $TumorRoot (Split-Path $_.Key -Leaf)
    }
}

$Plan | Export-Csv -LiteralPath $PlanPath -NoTypeInformation -Encoding UTF8

$TotalGiB = ($Plan | Measure-Object -Property SizeBytes -Sum).Sum / 1GB
$FreeGiB = (Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($RepositoryRoot).Substring(0,1))).Free / 1GB

[PSCustomObject]@{
    PlannedSlides = $Plan.Count
    NormalSlides = @($SelectedNormal).Count
    TumorSlides = @($SelectedTumor).Count
    PlannedGiB = [math]::Round($TotalGiB, 2)
    FreeGiB = [math]::Round($FreeGiB, 2)
    PlanPath = $PlanPath
} | Format-List

if ($InventoryOnly) {
    Write-Host "PASS: Inventory and download plan created; no slides downloaded." -ForegroundColor Green
    exit 0
}

if ($FreeGiB -lt ($TotalGiB + 20)) {
    throw "Insufficient free space. Keep at least 20 GiB beyond the planned download size."
}

foreach ($Row in $Plan) {
    if (Test-Path -LiteralPath $Row.Destination -PathType Leaf) {
        Write-Host "Already present: $($Row.Destination)" -ForegroundColor Yellow
        continue
    }
    Write-Host "Downloading $($Row.Key)" -ForegroundColor Cyan
    aws s3 cp "s3://camelyon-dataset/$($Row.Key)" $Row.Destination --no-sign-request --only-show-errors
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $($Row.Key)"
    }
    $ActualSize = (Get-Item -LiteralPath $Row.Destination).Length
    if ($ActualSize -ne [int64]$Row.SizeBytes) {
        throw "Size mismatch after download: $($Row.Destination)"
    }
}

Write-Host "PASS: Larger CAMELYON16 cohort downloaded and size-validated." -ForegroundColor Green
