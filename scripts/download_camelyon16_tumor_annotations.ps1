param(
    [string]$RepositoryRoot = "D:\HistoMetPath\HistoMetPath-repo"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    throw "AWS CLI is required."
}

$AnnotationRoot = Join-Path $RepositoryRoot "data\camelyon16\annotations"
$ManifestPath = Join-Path $AnnotationRoot "annotation_download_manifest.csv"
New-Item -ItemType Directory -Path $AnnotationRoot -Force | Out-Null

$RequiredSlides = @()
$RequiredSlides += 1..20 | ForEach-Object { "tumor_{0:D3}" -f $_ }
$RequiredSlides += "tumor_100"

$Rows = foreach ($Slide in $RequiredSlides) {
    $Key = "CAMELYON16/annotations/$Slide.xml"
    $Destination = Join-Path $AnnotationRoot "$Slide.xml"
    [PSCustomObject]@{
        Slide = $Slide
        Key = $Key
        Destination = $Destination
    }
}

foreach ($Row in $Rows) {
    if (Test-Path -LiteralPath $Row.Destination -PathType Leaf) {
        Write-Host "Already present: $($Row.Slide)" -ForegroundColor Yellow
        continue
    }

    Write-Host "Downloading $($Row.Key)" -ForegroundColor Cyan
    aws s3 cp `
        "s3://camelyon-dataset/$($Row.Key)" `
        $Row.Destination `
        --no-sign-request `
        --only-show-errors

    if ($LASTEXITCODE -ne 0) {
        throw "Annotation download failed: $($Row.Key)"
    }

    if ((Get-Item -LiteralPath $Row.Destination).Length -le 0) {
        throw "Downloaded annotation is empty: $($Row.Destination)"
    }
}

$Rows | ForEach-Object {
    $Item = Get-Item -LiteralPath $_.Destination
    [PSCustomObject]@{
        Slide = $_.Slide
        Key = $_.Key
        Destination = $_.Destination
        SizeBytes = $Item.Length
        SHA256 = (Get-FileHash -LiteralPath $_.Destination -Algorithm SHA256).Hash.ToLowerInvariant()
    }
} | Export-Csv -LiteralPath $ManifestPath -NoTypeInformation -Encoding UTF8

Write-Host "PASS: CAMELYON16 tumor annotations downloaded and checksummed." -ForegroundColor Green
