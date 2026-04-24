; Inno Setup script for Star Trail CleanR
; Compiled in CI by ISCC.exe with /DAppVersion=<version> passed in.
;
; Why this exists:
; Shipping the PyInstaller --onedir build as a raw zip forces Windows users to
; extract 60,000+ small files, which Windows Explorer takes hours to do. This
; script wraps the same bundle in a one-click installer so users get a normal
; Setup.exe experience: double-click, Next, Install, done.

#ifndef AppVersion
  #define AppVersion "0.000"
#endif

; Override via /DOutputName=StarTrailCleanRSetup-NVIDIA (or similar) to produce
; variant installers without duplicating this entire .iss file.
#ifndef OutputName
  #define OutputName "StarTrailCleanRSetup"
#endif

#define AppName "Star Trail CleanR"
#define AppPublisher "Bruce Herwig"
#define AppURL "https://github.com/bruceherwig-dot/StarTrailCleanR"
#define AppExeName "StarTrailCleanR.exe"

[Setup]
; Stable AppId — required so future installers recognize an existing install
; and upgrade in place instead of side-by-side. Do not regenerate.
AppId={{6E3D8E62-7B4F-4C0E-A9B1-1F2D5D7C5C90}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
DefaultDirName={autopf}\StarTrailCleanR
DefaultGroupName=Star Trail CleanR
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename={#OutputName}
SetupIconFile=..\assets\StarTrailCleanR.ico
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "..\dist\StarTrailCleanR\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Star Trail CleanR"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall Star Trail CleanR"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Star Trail CleanR"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch Star Trail CleanR"; Flags: nowait postinstall skipifsilent
