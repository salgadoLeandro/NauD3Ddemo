param([string]$file="")

Add-Type -AssemblyName Microsoft.VisualBasic
Add-Type -AssemblyName System.Windows.Forms

if ($file -eq "") {
    Write-Output "Nothing here"
}
else {
    $sep = "|"
    $options = [System.StringSplitOptions]::RemoveEmptyEntries

    [string[]]$inputs = Get-Content -Path $PSScriptRoot"\"$file
    
    md -Force $PSScriptRoot"\output" | Out-Null
    
    for ($i=0; $i -lt $inputs.Length; $i++) {
        [string[]]$splits = $inputs[$i].Split($sep, $options)
        $filename = $splits[0]
        $outname = $splits[1]

        $out = "Starting " + $filename + "..."
        Write-Output $out

        for ($k=0; $k -lt 5; $k++) {

            $outfile = $PSScriptRoot + "\output\out_" + $outname + "_" + $k + ".txt"
            $proc = (Start-Process -FilePath $PSScriptRoot"\nauD3DDemod.exe" -ArgumentList $filename -PassThru -RedirectStandardOutput $outfile)

            Start-Sleep -Seconds 5
            [System.Windows.Forms.SendKeys]::SendWait("R")
            Start-Sleep -Seconds 1
            [System.Windows.Forms.SendKeys]::SendWait("P")
            Start-Sleep -Seconds 1
            Stop-Process $proc

            Start-Sleep -Seconds 1
            (Get-Content -Path $outfile).Where({ $_ -like "Name*" }, 'SkipUntil') | Out-File -FilePath $outfile
        }
        
        Write-Output "Done"
    }
}