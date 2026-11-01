param([string]$JobDir)
$task = Get-ScheduledTask -TaskName 'keyframes-job' -ErrorAction SilentlyContinue
$state = if ($task) { $task.State } else { 'none' }
$done = Test-Path "$JobDir\out\done.txt"
$have = if (Test-Path "$JobDir\in") { (Get-ChildItem "$JobDir\in" -Name) -join ',' } else { '' }
Write-Output "TASK $state"
Write-Output "DONE $done"
Write-Output "HAVE $have"
