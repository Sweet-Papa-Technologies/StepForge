<#
.SYNOPSIS
    Authenticode-sign one Windows file with Azure Trusted Signing.

.DESCRIPTION
    Tauri calls this once per file it produces (the app .exe and the NSIS
    installer) through `bundle.windows.signCommand`, which the release workflow
    sets. It uses Microsoft's `sign` tool and whatever identity the Azure CLI is
    logged in as: in CI that is the GitHub OIDC login (azure/login), locally an
    `az login`. No certificate, PFX or secret is stored anywhere.

    Account details (not secret) come from desktop/metadata.json:
      Endpoint, CodeSigningAccountName, CertificateProfileName

    One-time setup on a machine:
      az login
      dotnet tool install --global sign --prerelease

.EXAMPLE
    pwsh -NoProfile -File scripts/sign-windows.ps1 -Path .\StepForge.exe
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Path,
    [string]$Description = 'StepForge'
)

$ErrorActionPreference = 'Stop'
$appDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$meta = Get-Content (Join-Path $appDir 'metadata.json') -Raw | ConvertFrom-Json
$file = (Resolve-Path $Path).Path

$sign = Get-Command sign -ErrorAction SilentlyContinue
$signExe = if ($sign) { $sign.Source } else { Join-Path $HOME '.dotnet\tools\sign.exe' }
if (-not (Test-Path $signExe)) {
    throw "the 'sign' tool is not installed: dotnet tool install --global sign --prerelease"
}

Write-Host "[sign-windows] signing $file"
& $signExe code trusted-signing (Split-Path -Leaf $file) `
    --base-directory (Split-Path -Parent $file) `
    --trusted-signing-endpoint $meta.Endpoint `
    --trusted-signing-account $meta.CodeSigningAccountName `
    --trusted-signing-certificate-profile $meta.CertificateProfileName `
    --description $Description `
    --description-url 'https://github.com/Sweet-Papa-Technologies/StepForge' `
    --timestamp-url 'http://timestamp.acs.microsoft.com' `
    --file-digest SHA256 `
    --verbosity Warning
if ($LASTEXITCODE -ne 0) { throw "sign exited with $LASTEXITCODE for $file" }

$sig = Get-AuthenticodeSignature $file
Write-Host "[sign-windows] $($sig.Status): $($sig.SignerCertificate.Subject)"
if ($sig.Status -ne 'Valid') { throw "signature on $file is $($sig.Status)" }
