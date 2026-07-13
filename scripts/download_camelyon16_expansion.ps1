param(
    [string]$RepositoryRoot = "D:\HistoMetPath\HistoMetPath-repo",
    [int]$FirstSlideNumber = 11,
    [int]$LastSlideNumber = 20,
    [switch]$InventoryOnly
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    throw "AWS CLI is required."
}
if ($FirstSlideNumber -gt $LastSlideNumber) {
    throw "FirstSlideNumber must be less than or equal to LastSlideNumber."
}

$DataRoot = Join-Path $RepositoryRoot "data\camelyon16"
$NormalRoot = Join-Path $DataRoot "training\normal"
$TumorRoot = Join-Path $DataRoot "training\tumor"
$InventoryPath = Join-Path $DataRoot "aws_inventory.txt"
$PlanPath = Join-Path $DataRoot "expansion_download_plan.csv"

New-Item -ItemType Directory -Path $NormalRoot -Force | Out-Null
New-Item -ItemType Directory -Path $TumorRoot -Force | Out-Null

Write-Host "Building CAMELYON16 public S3 inventory..." -ForegroundColor Cyan
aws s3 ls s3://camelyon-dataset/CAMELYON16/ --recursive --no-sign-request |
    Set-Content -LiteralPath $InventoryPath -Encoding UTF8

$InventoryRows = Get-Content -LiteralPath $InventoryPath | ForEach-Object {
    if ($_ -match '^\s*(\S+)\s+(\S+)\s+(\d+)\s+(.+)$') {
        [PSCustomObject]@{
            SizeBytes = [int64]$Matches[3]
            Key = $Matches[4]
        }
    }
}

$ByKey = @{}
foreach ($Row in $InventoryRows) {
    $ByKey[$Row.Key] = $Row
}

$Plan = @()
foreach ($ClassName in @("normal", "tumor")) {
    foreach ($Number in $FirstSlideNumber..$LastSlideNumber) {
        $FileName = "{0}_{1:D3}.tif" -f $ClassName, $Number
        $Key = "CAMELYON16/images/$FileName"
        if (-not $ByKey.ContainsKey($Key)) {
            throw "Required S3 key not found: $Key"
        }
        $DestinationRoot = if ($ClassName -eq "normal") { $NormalRoot } else { $TumorRoot }
        $Plan += [PSCustomObject]@{
            Class = $ClassName
            Slide = [System.IO.Path]::GetFileNameWithoutExtension($FileName)
            Key = $Key
            SizeBytes = [int64]$ByKey[$Key].SizeBytes
            Destination = Join-Path $DestinationRoot $FileName
        }
    }
}

$Plan | Export-Csv -LiteralPath $PlanPath -NoTypeInformation -Encoding UTF8
$PlannedBytes = [double](($Plan | Measure-Object -Property SizeBytes -Sum).Sum)
$DriveName = [System.IO.Path]::GetPathRoot($RepositoryRoot).Substring(0, 1)
$FreeBytes = [double](Get-PSDrive -Name $DriveName).Free
$RequiredBytes = $PlannedBytes + 20GB

[PSCustomObject]@{
    PlannedSlides = $Plan.Count
    NormalSlides = @($Plan | Where-Object Class -eq "normal").Count
    TumorSlides = @($Plan | Where-Object Class -eq "tumor").Count
    PlannedGiB = "{0:F2}" -f ($PlannedBytes / 1GB)
    FreeGiB = "{0:F2}" -f ($FreeBytes / 1GB)
    RequiredGiB = "{0:F2}" -f ($RequiredBytes / 1GB)
    EnoughSpace = $FreeBytes -ge $RequiredBytes
    PlanPath = $PlanPath
} | Format-List

if ($InventoryOnly) {
    Write-Host "PASS: Expansion inventory created; no slides downloaded." -ForegroundColor Green
    exit 0
}
if ($FreeBytes -lt $RequiredBytes) {
    throw "Insufficient free space for expansion plus 20 GiB reserve."
}

foreach ($Row in $Plan) {
    if (Test-Path -LiteralPath $Row.Destination -PathType Leaf) {
        $ExistingSize = (Get-Item -LiteralPath $Row.Destination).Length
        if ($ExistingSize -ne [int64]$Row.SizeBytes) {
            throw "Existing file has incorrect size: $($Row.Destination)"
        }
        Write-Host "Already present and size-valid: $($Row.Slide)" -ForegroundColor Yellow
        continue
    }
    Write-Host "Downloading $($Row.Key)" -ForegroundColor Cyan
    aws s3 cp "s3://camelyon-dataset/$($Row.Key)" $Row.Destination --no-sign-request --only-show-errors
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $($Row.Key)"
    }
    $ActualSize = (Get-Item -LiteralPath $Row.Destination).Length
    if ($ActualSize -ne [int64]$Row.SizeBytes) {
        throw "Downloaded file size mismatch: $($Row.Destination)"
    }
}

Write-Host "PASS: Fresh-holdout expansion cohort downloaded and size-validated." -ForegroundColor Green
