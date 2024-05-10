# -*- coding: utf-8 -*-
"""
| ----------------------------------------------------------------------------------------------------------------------
| Date                : February 2020                                todo: change date, copyright and email in all files
| Copyright           : © 2020 by Benjamin Jakimow and Ann Crabbé
| Email               : acrabbe.foss@gmail.com
| Acknowledgements    : Based on 'Create A QGIS Plugin' [https://bitbucket.org/kul-reseco/create-qgis-plugin]
|                       Crabbé Ann, Jakimow Benjamin, Somers Ben; funded by BELSPO STEREO III (LUMOS - SR/01/321)
|
| This file is part of the [INSERT PLUGIN NAME] plugin and [INSERT PYTHON PACKAGE NAME] python package.
| todo: make sure to fill out your plugin name and python package name here.
|
| This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
| License as published by the Free Software Foundation, either version 3 of the License, or any later version.
|
| This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
| warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
|
| You should have received a copy of the GNU General Public License (COPYING.txt). If not see www.gnu.org/licenses.
| ----------------------------------------------------------------------------------------------------------------------
"""
import re
import os
import sys
import pathlib
import zipfile
import typing

from create_resource_file import compile_plugin_resources as build_resources
import package_variables


class QGISMetadataFileWriter(object):

    def __init__(self, name, qgis_min_version, description, version, author, about, email, repository,
                 qgis_max_version=None, changelog=None, is_experimental=None, is_deprecated=None, tags=None,
                 homepage=None, tracker=None, icon=None, category=None, has_processing_provider=None):

        # required parameters
        self._name = name  # name of the plugin
        self._qgis_min_version = qgis_min_version  # dotted notation of minimum QGIS version
        self._description = description  # short description, no HTML allowed
        self._version = version  # dotted notation
        self._author = author  # author name
        self._about = about  # longer description, no HTML allowed
        self._email = email  # email of the author (only visible to other plugin authors)
        self._repository = repository  # source code repository (must be a valid url)
        self._tracker = tracker  # valid URL for tickets and bug reports

        # optional parameters
        self._qgis_max_version = qgis_max_version  # dotted notation of maximum QGIS version
        self._changelog = changelog  # string, can be multi line, no HTML allowed
        self._is_experimental = is_experimental  # boolean
        self._is_deprecated = is_deprecated  # boolean, applies to the whole plugin and not just the version
        self._tags = tags  # comma separated list, spaces are allowed inside individual tags
        self._homepage = homepage  # valid URL pointing to the homepage of your plugin
        self._icon = icon  # file name or relative path (to the base folder): PNG, JPEG
        self._category = category  # one of Raster, Vector, Database and Web
        self._has_processing_provider = has_processing_provider  # boolean

    def metadataString(self) -> str:
        lines = ['[general]',
                 'name={}'.format(self._name),
                 'qgisMinimumVersion={}'.format(self._qgis_min_version),
                 'description={}'.format(self._description),
                 'version={}'.format(self._version),
                 'author={}'.format(self._author),
                 'about={}'.format(re.sub('\n', '', self._about)),
                 'email={}'.format(self._email),
                 'tracker={}'.format(self._tracker),
                 'repository={}'.format(self._repository)
                 ]
        if self._qgis_max_version:
            lines.append('qgisMaximumVersion={}'.format(self._qgis_max_version))
        if self._changelog:
            lines.append('changelog={}'.format(self._changelog))
        if self._is_experimental:
            lines.append('experimental=True')
        if self._is_deprecated:
            lines.append('deprecated=True')
        if self._tags:
            lines.append('tags={}'.format(', '.join(self._tags)))
        if self._homepage:
            lines.append('homepage={}'.format(self._homepage))
        if self._icon:
            lines.append('icon={}'.format(self._icon))
        if self._category:
            lines.append('category={}'.format(self._category))
        if self._has_processing_provider:
            lines.append('hasProcessingProvider=yes')

        return '\n'.join(lines)

    def writeMetadataTxt(self, metadata_path: str):
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(self.metadataString())


def scan_tree(scan_path, ending='') -> typing.Iterator[pathlib.Path]:
    """
    Recursively returns file paths in directory
    :param scan_path: root directory to search in
    :param ending: str with required file ending, e.g. ".py" to search for *.py files
    :return: pathlib.Path
    """
    for entry in os.scandir(scan_path):
        if entry.is_dir(follow_symlinks=False):
            yield from scan_tree(entry.path, ending=ending)
        elif entry.is_file and entry.path.endswith(ending):
            yield pathlib.Path(entry.path)


def create_plugin_zip():
    """ Create the plugin.zip """

    metadata = QGISMetadataFileWriter(
        name=package_variables.long_name,
        qgis_min_version=package_variables.qgis_min_version,
        description=package_variables.short_description,
        version=package_variables.long_version,
        author=package_variables.author,
        about=package_variables.long_description,
        email=package_variables.author_email,
        repository=package_variables.bitbucket_src,
        qgis_max_version=None,
        changelog=None,
        is_experimental=None,
        is_deprecated=None,
        tags=package_variables.keywords,
        homepage=package_variables.read_the_docs,
        tracker=package_variables.bitbucket_issues,
        icon=package_variables.qgis_metadata_icon,
        category=package_variables.qgis_category,
        has_processing_provider=package_variables.processing_provider)

    repository_folder = pathlib.Path(__file__).parents[0].resolve()
    # todo: change this if you have changed the qgis plugin folder name
    plugin_directory = repository_folder / 'qgisplugin'
    metadata.writeMetadataTxt(plugin_directory / 'metadata.txt')

    distribution_path = repository_folder / 'dist' / '{}-{}-qgis.zip'.format(package_variables.dense_name,
                                                                             package_variables.long_version)
    os.makedirs(distribution_path.parent, exist_ok=True)

    allowed_endings = re.compile(r'\.(py|txt|md|png|svg|ui)$')
    plugin_files = [f for f in scan_tree(plugin_directory) if allowed_endings.search(f.as_posix())]

    for file in plugin_files:
        assert file.is_file(), 'File does not exists: {}'.format(file)

    # create the zip file that contains all plugin_files and can be installed with the QGIS Plugin Manager
    with zipfile.ZipFile(distribution_path, 'w') as f:
        for file in plugin_files:
            archive_name = file.relative_to(repository_folder).as_posix()
            f.write(file, arcname=archive_name)

    print('QGIS Plugin created: {}'.format(distribution_path.as_posix()))


if __name__ == "__main__":

    import getopt

    try:
        print(sys.argv)
        opts, options = getopt.getopt(sys.argv[1:], "")
    except getopt.GetoptError as err:
        print(err)

    build_resources()
    create_plugin_zip()
    exit(0)
