$Project = Split-Path -Parent $PSScriptRoot
$Action = New-ScheduledTaskAction -Execute "$Project\run_full_pipeline.bat" -WorkingDirectory $Project
$Trigger = New-ScheduledTaskTrigger -Daily -At 9:00am
Register-ScheduledTask -TaskName "IA Paris Sportifs - Update quotidien" -Action $Action -Trigger $Trigger -Description "Met à jour les prédictions de paris sportifs tous les jours" -Force
Write-Host "Tache planifiee creee : IA Paris Sportifs - Update quotidien"
