param([string]$JobDir)
New-Item -ItemType Directory -Force -Path "$JobDir\in", "$JobDir\out" | Out-Null
$task = Get-ScheduledTask -TaskName 'keyframes-job' -ErrorAction SilentlyContinue
if ($task -and $task.State -eq 'Running') { Write-Output 'STATE running'; exit 0 }
if (Test-Path "$JobDir\out\done.txt") { Write-Output 'STATE done'; exit 0 }
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File C:\sam3\keyframes\run_keyframes.ps1 -JobDir $JobDir"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddYears(10)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName 'keyframes-job' -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName 'keyframes-job'
Write-Output 'STATE started'
