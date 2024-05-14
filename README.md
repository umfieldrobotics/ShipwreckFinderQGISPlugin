# Shipwreck Seeker QGIS Plugin

This [QGIS](https://www.qgis.org/en/site/) plugin finds shipwrecks from multibeam sonar data. 

*The project structure is based on 'Create A QGIS Plugin' created by Crabbé Ann, Jakimow Benjamin*
*and Somers Ben and funded by BELSPO STEREO III (Project LUMOS - SR/01/321).*
*The full code is available from https://bitbucket.org/kul-reseco/create-qgis-plugin.*

### Usage
Todo todo

Need to install torch and stuff like this (if on mac): 
```bash
/Applications/QGIS.app/Contents/MacOS/bin/pip3 install --upgrade remotior-sensus scikit-learn torch
```

Then you have to run the following which will create a new zip file in ./dist that can be used to install the package in QGIS

**For Linux**
Install QGIS from terminal...
```bash
sudo su
apt install python3-pip
pip3 install torch
```

```bash
python3 create_plugin.py
```

### SOFTWARE LICENSE

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License (COPYING.txt). If not see www.gnu.org/licenses.

