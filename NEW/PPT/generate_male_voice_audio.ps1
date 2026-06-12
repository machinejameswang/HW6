param(
    [string]$VoiceName = "Microsoft David Desktop",
    [int]$Rate = -1,
    [int]$Volume = 95
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = Join-Path $scriptDir "hw6_20_slide_voiceover_script.txt"
$audioDir = Join-Path $scriptDir "audio"
$manifestPath = Join-Path $audioDir "audio_manifest.csv"

New-Item -ItemType Directory -Force -Path $audioDir | Out-Null

Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice($VoiceName)
$synth.Rate = $Rate
$synth.Volume = $Volume

$content = Get-Content -Raw -Encoding UTF8 $scriptPath
$pattern = '(?ms)^Slide\s+(\d+):\s*(.+?)\r?\n(.*?)(?=^Slide\s+\d+:|\z)'
$matches = [regex]::Matches($content, $pattern)

$manifest = New-Object System.Collections.Generic.List[string]
$manifest.Add("slide,title,audio_file,voice,rate,volume")

foreach ($match in $matches) {
    $slide = [int]$match.Groups[1].Value
    $title = $match.Groups[2].Value.Trim()
    $text = $match.Groups[3].Value.Trim()
    $fileName = "slide_{0:D2}_male_narration.wav" -f $slide
    $outPath = Join-Path $audioDir $fileName

    $synth.SetOutputToWaveFile($outPath)
    $synth.Speak($text)
    $synth.SetOutputToNull()

    $safeTitle = $title.Replace('"', '""')
    $manifest.Add("$slide,""$safeTitle"",$fileName,""$VoiceName"",$Rate,$Volume")
}

$manifest | Set-Content -Encoding UTF8 $manifestPath
$synth.Dispose()

Write-Host "Generated $($matches.Count) narration audio files."
Write-Host "Audio directory: $audioDir"
Write-Host "Manifest: $manifestPath"
