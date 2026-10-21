# PyInstaller spec — macOS 13+ on Apple Silicon (dist/DefectLab.app; build_mac.sh also makes the .dmg).
# Build: packaging/build_mac.sh  (or: python -m PyInstaller --noconfirm packaging/defectlab_mac.spec)
# One executable inside the bundle: the training worker is the same program started with
# --train-worker (stdout works normally on macOS, unlike a windowed Windows exe).
import sys

sys.path.insert(0, SPECPATH)
sys.path.insert(0, str(__import__("pathlib").Path(SPECPATH).parent))
from spec_common import APP_NAME, BUNDLE_ID, collect  # noqa: E402

from defect_inspection import __version__  # noqa: E402

c = collect(SPECPATH)
a = Analysis([c["app_script"]], **c["analysis"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=APP_NAME, console=False, strip=False, upx=False,
          target_arch="arm64", codesign_identity=None, entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=APP_NAME)
app = BUNDLE(
    coll,
    name=f"{APP_NAME}.app",
    bundle_identifier=BUNDLE_ID,
    version=__version__,
    info_plist={
        "CFBundleDisplayName": APP_NAME,
        "CFBundleShortVersionString": __version__,
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,  # follow light/dark mode
        "LSApplicationCategoryType": "public.app-category.developer-tools",
    },
)
