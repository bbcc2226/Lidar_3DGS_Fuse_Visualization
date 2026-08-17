#!/usr/bin/env bash
set -euo pipefail

echo "Installing/checking C++ dependencies..."

if [[ ! -r /etc/os-release ]]; then
    echo "[ERROR] /etc/os-release is unavailable; this installer supports Ubuntu 20.04 and 24.04."
    exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "[WARNING] Detected ${PRETTY_NAME:-an unknown distribution}."
    echo "          Package names below are tested on Ubuntu 20.04 and 24.04."
elif [[ "${VERSION_ID:-}" != "20.04" && "${VERSION_ID:-}" != "24.04" ]]; then
    echo "[WARNING] Detected Ubuntu ${VERSION_ID:-unknown}; expected 20.04 or 24.04."
fi

sudo apt-get update

PACKAGES=(
    build-essential
    cmake
    git
    pkg-config
    libopencv-dev
    libeigen3-dev
    libceres-dev
    libsuitesparse-dev
    libgoogle-glog-dev
    libgflags-dev
    libgtest-dev
    libyaml-cpp-dev
    nlohmann-json3-dev
    # Do not use the removed qt5-default metapackage. These packages provide
    # Qt Core, Gui, Widgets, qmake, moc, uic and rcc on both supported Ubuntu
    # releases. Ubuntu 20.04 supplies Qt 5.12; 24.04 supplies a newer Qt 5,
    # which remains compatible with a project requiring Qt >= 5.12.
    qtbase5-dev
    qtbase5-dev-tools
    qt5-qmake
    qtchooser
    libqt5opengl5-dev
)

for package in "${PACKAGES[@]}"; do
    if dpkg -s "${package}" >/dev/null 2>&1; then
        echo "[OK] ${package} already installed"
    else
        echo "[INSTALL] ${package}"
        sudo apt-get install -y "${package}"
    fi
done

echo
echo "Verifying dependencies..."

cmake --version | head -n 1

if pkg-config --exists opencv4; then
    echo "[OK] OpenCV $(pkg-config --modversion opencv4)"
else
    echo "[ERROR] OpenCV was not detected by pkg-config"
    exit 1
fi

if [[ -d /usr/include/eigen3/Eigen ]]; then
    echo "[OK] Eigen found at /usr/include/eigen3"
else
    echo "[ERROR] Eigen headers not found"
    exit 1
fi

echo
echo "Checking Ceres..."

if ! dpkg-query -W -f='${Status}' libceres-dev 2>/dev/null \
    | grep -q "install ok installed"; then
    echo "[ERROR] libceres-dev is not installed"
    exit 1
fi

echo "[OK] libceres-dev is installed"

echo
echo "Checking Qt..."

QT_MINIMUM_VERSION="5.12.0"
QT_MODULES=(Qt5Core Qt5Gui Qt5Widgets)
for module in "${QT_MODULES[@]}"; do
    if ! pkg-config --exists "${module}"; then
        echo "[ERROR] ${module} was not detected by pkg-config"
        exit 1
    fi
    echo "[OK] ${module} $(pkg-config --modversion "${module}")"
done

QT_VERSION="$(pkg-config --modversion Qt5Core)"
if ! dpkg --compare-versions "${QT_VERSION}" ge "${QT_MINIMUM_VERSION}"; then
    echo "[ERROR] Qt ${QT_VERSION} is installed, but Qt ${QT_MINIMUM_VERSION} or newer is required"
    exit 1
fi

QMAKE_BIN="$(command -v qmake || command -v qmake5 || true)"
if [[ -z "${QMAKE_BIN}" ]]; then
    echo "[ERROR] Qt 5 qmake was not found"
    exit 1
fi

QMAKE_QT_VERSION="$("${QMAKE_BIN}" -query QT_VERSION 2>/dev/null || true)"
if [[ "${QMAKE_QT_VERSION}" != 5.* ]]; then
    echo "[ERROR] ${QMAKE_BIN} does not select Qt 5 (reported: ${QMAKE_QT_VERSION:-unknown})"
    echo "        Select Qt 5 with: sudo update-alternatives --config qmake"
    exit 1
fi

echo "[OK] Qt ${QT_VERSION} satisfies the Qt >= ${QT_MINIMUM_VERSION} requirement"
echo "[OK] qmake: ${QMAKE_BIN} (Qt ${QMAKE_QT_VERSION})"

CERES_HEADER="$(find /usr/include /usr/local/include \
    -path '*/ceres/ceres.h' \
    -print -quit 2>/dev/null || true)"

if [[ -z "${CERES_HEADER}" ]]; then
    echo "[ERROR] ceres/ceres.h was not found"
    exit 1
fi

echo "[OK] Ceres header found:"
echo "     ${CERES_HEADER}"

CERES_CONFIG="$(find /usr /usr/local \
    -name 'CeresConfig.cmake' \
    -print -quit 2>/dev/null || true)"

if [[ -z "${CERES_CONFIG}" ]]; then
    echo "[ERROR] CeresConfig.cmake was not found"
    exit 1
fi

echo "[OK] Ceres CMake configuration found:"
echo "     ${CERES_CONFIG}"

CERES_LIBRARY="$(find /usr/lib /usr/local/lib \
    \( -name 'libceres.so*' -o -name 'libceres.a' \) \
    -print -quit 2>/dev/null || true)"

if [[ -n "${CERES_LIBRARY}" ]]; then
    echo "[OK] Ceres library found:"
    echo "     ${CERES_LIBRARY}"
else
    echo "[WARNING] Ceres library file was not found by filesystem search"
    echo "          The compile test below will determine whether Ceres is usable."
fi

echo
echo "Installed package versions:"
dpkg-query -W -f='OpenCV package: ${Version}\n' libopencv-dev
dpkg-query -W -f='Eigen package:  ${Version}\n' libeigen3-dev
dpkg-query -W -f='Ceres package:  ${Version}\n' libceres-dev
dpkg-query -W -f='Qt base package: ${Version}\n' qtbase5-dev

echo
echo "All required libraries are installed and available."
echo "You can build the project separately with:"
echo "  cmake -S . -B build"
echo "  cmake --build build -j$(nproc)"
