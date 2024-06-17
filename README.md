# Shipwreck Seeker QGIS Plugin

This [QGIS](https://www.qgis.org/en/site/) plugin finds shipwrecks from multibeam sonar data. 

*The project structure is based on 'Create A QGIS Plugin' created by Crabbé Ann, Jakimow Benjamin*
*and Somers Ben and funded by BELSPO STEREO III (Project LUMOS - SR/01/321).*
*The full code is available from https://bitbucket.org/kul-reseco/create-qgis-plugin.*

### Installation
**OPTION 1: With Conda (recommended)**
1. Create a new conda env
```bash
conda create --name qgis_env python=3.11
conda activate qgis_env
```
2. Install QGIS
```bash
conda install -c conda-forge qgis
```
3. Clone this repository
```bash
git clone git@github.com:umfieldrobotics/ShipwreckSeekerQGISPlugin.git
```
3. Install requirements
```bash
pip install -r requirements.txt
```

4. Run qgis
```bash
qgis
```

**OPTION 2: Install ontop of previous QGIS Installation (Mac)**

Locate the qgis version of pip and install dependencies
```bash
/Applications/QGIS.app/Contents/MacOS/bin/pip3 install --upgrade -r requirements.txt
```

**OPTION 3: Install ontop of previous QGIS Installation (Linux)**
Install QGIS from terminal...
```bash
sudo su
apt install python3-pip
pip3 install -r requirements.txt
```

## Build the Plugin
```bash
python3 create_plugin.py
```

This should create a tar.gz file in the `./dist` folder. To load the plugin, go to `Plugins` -> `Manage and Install Plugins...` -> `Install from ZIP` and input the file that was just created. Then go to the  `Installed` tab, and toggle the `Shipwreck Seeker Plugin` off and on to activate it. 

## Usage
To use the plugin, go to `Raster` -> `Shipwreck Seeker Menu` -> `Shipwreck Seeker`.

### SOFTWARE LICENSE

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License (COPYING.txt). If not see www.gnu.org/licenses.

