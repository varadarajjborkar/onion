param([string]$JobDir)
$py = 'C:\sam3\.venv\Scripts\python.exe'
"START $((Get-Date).ToString('s'))" | Add-Content "$JobDir\log.txt"
& $py -u C:\sam3\keyframes\run_keyframes.py $JobDir *>> "$JobDir\log.txt"
"EXIT $LASTEXITCODE  $((Get-Date).ToString('s'))" | Add-Content "$JobDir\log.txt"
