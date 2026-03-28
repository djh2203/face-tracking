$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$models = Join-Path $root "models"
New-Item -ItemType Directory -Force -Path $models | Out-Null

$protoPath = Join-Path $models "deploy.prototxt"
$modelPath = Join-Path $models "res10_300x300_ssd_iter_140000.caffemodel"

$protoUrls = @(
  "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
  "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt"
)

$modelUrls = @(
  "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn_samples_face_detector/res10_300x300_ssd_iter_140000.caffemodel",
  "https://github.com/opencv/opencv_3rdparty/raw/master/dnn_samples_face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)

function Download-FirstOk($urls, $outPath) {
  foreach ($u in $urls) {
    try {
      Write-Host "Downloading $u"
      Invoke-WebRequest -Uri $u -OutFile $outPath -UseBasicParsing
      if ((Get-Item $outPath).Length -gt 0) { return }
    } catch {
      Write-Host "Failed: $u"
    }
  }
  throw "All download URLs failed for $outPath"
}

if (-not (Test-Path $protoPath) -or (Get-Item $protoPath).Length -eq 0) {
  Download-FirstOk $protoUrls $protoPath
} else {
  Write-Host "OK: $protoPath"
}

if (-not (Test-Path $modelPath) -or (Get-Item $modelPath).Length -eq 0) {
  Download-FirstOk $modelUrls $modelPath
} else {
  Write-Host "OK: $modelPath"
}

Write-Host "Done."
