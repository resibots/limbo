#!/usr/bin/env python
# encoding: utf-8
#| Copyright Inria May 2015
#| This project has received funding from the European Research Council (ERC) under
#| the European Union's Horizon 2020 research and innovation programme (grant
#| agreement No 637972) - see http://www.resibots.eu
#|
#| Contributor(s):
#|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
#|   - Antoine Cully (antoinecully@gmail.com)
#|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
#|   - Federico Allocati (fede.allocati@gmail.com)
#|   - Vaios Papaspyros (b.papaspyros@gmail.com)
#|   - Roberto Rama (bertoski@gmail.com)
#|
#| This software is a computer library whose purpose is to optimize continuous,
#| black-box functions. It mainly implements Gaussian processes and Bayesian
#| optimization.
#| Main repository: http://github.com/resibots/limbo
#| Documentation: http://www.resibots.eu/limbo
#|
#| This software is governed by the CeCILL-C license under French law and
#| abiding by the rules of distribution of free software.  You can  use,
#| modify and/ or redistribute the software under the terms of the CeCILL-C
#| license as circulated by CEA, CNRS and INRIA at the following URL
#| "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and  rights to copy,
#| modify and redistribute granted by the license, users are provided only
#| with a limited warranty  and the software's author,  the holder of the
#| economic rights,  and the successive licensors  have only  limited
#| liability.
#|
#| In this respect, the user's attention is drawn to the risks associated
#| with loading,  using,  modifying and/or developing or reproducing the
#| software by the user in light of its specific status of free software,
#| that may mean  that it is complicated to manipulate,  and  that  also
#| therefore means  that it is reserved for developers  and  experienced
#| professionals having in-depth computer knowledge. Users are therefore
#| encouraged to load and test the software's suitability as regards their
#| requirements in conditions enabling the security of their systems and/or
#| data to be ensured and,  more generally, to use and operate it in the
#| same conditions as regards security.
#|
#| The fact that you are presently reading this means that you have had
#| knowledge of the CeCILL-C license and that you accept its terms.
#|
#! /usr/bin/env python
# XCode 3/XCode 4 generator for Waf
# Nicolas Mercier 2011

"""
Usage:

def options(opt):
    opt.load('xcode')

$ waf configure xcode
"""

# TODO: support iOS projects

from waflib import Context, TaskGen, Build, Utils
import os, sys, random, time

HEADERS_GLOB = '**/(*.h|*.hpp|*.H|*.inl)'

MAP_EXT = {
    '.h' :  "sourcecode.c.h",

    '.hh':  "sourcecode.cpp.h",
    '.inl': "sourcecode.cpp.h",
    '.hpp': "sourcecode.cpp.h",

    '.c':   "sourcecode.c.c",

    '.m':   "sourcecode.c.objc",

    '.mm':  "sourcecode.cpp.objcpp",

    '.cc':  "sourcecode.cpp.cpp",

    '.cpp': "sourcecode.cpp.cpp",
    '.C':   "sourcecode.cpp.cpp",
    '.cxx': "sourcecode.cpp.cpp",
    '.c++': "sourcecode.cpp.cpp",

    '.l':   "sourcecode.lex", # luthor
    '.ll':  "sourcecode.lex",

    '.y':   "sourcecode.yacc",
    '.yy':  "sourcecode.yacc",

    '.plist': "text.plist.xml",
    ".nib":   "wrapper.nib",
    ".xib":   "text.xib",
}

def newid():
    return "%04X%04X%04X%012d" % (random.randint(0, 32767), random.randint(0, 32767), random.randint(0, 32767), int(time.time()))

class XCodeNode:
    def __init__(self):
        self._id = newid()

    def tostring(self, value):
        if isinstance(value, dict):
            result = "{\n"
            for k,v in value.iteritems():
                result = result + "\t\t\t%s = %s;\n" % (k, self.tostring(v))
            result = result + "\t\t}"
            return result
        elif isinstance(value, str):
            return "\"%s\"" % value
        elif isinstance(value, list):
            result = "(\n"
            for i in value:
                result = result + "\t\t\t%s,\n" % self.tostring(i)
            result = result + "\t\t)"
            return result
        elif isinstance(value, XCodeNode):
            return value._id
        else:
            return str(value)

    def write_recursive(self, value, file):
        if isinstance(value, dict):
            for k,v in value.iteritems():
                self.write_recursive(v, file)
        elif isinstance(value, list):
            for i in value:
                self.write_recursive(i, file)
        elif isinstance(value, XCodeNode):
            value.write(file)

    def write(self, file):
        for attribute,value in self.__dict__.iteritems():
            if attribute[0] != '_':
                self.write_recursive(value, file)

        w = file.write
        w("\t%s = {\n" % self._id)
        w("\t\tisa = %s;\n" % self.__class__.__name__)
        for attribute,value in self.__dict__.iteritems():
            if attribute[0] != '_':
                w("\t\t%s = %s;\n" % (attribute, self.tostring(value)))
        w("\t};\n\n")



# Configurations
class XCBuildConfiguration(XCodeNode):
    def __init__(self, name, settings = {}):
        XCodeNode.__init__(self)
        self.baseConfigurationReference = ""
        self.buildSettings = settings
        self.name = name

class XCConfigurationList(XCodeNode):
    def __init__(self, settings):
        XCodeNode.__init__(self)
        self.buildConfigurations = settings
        self.defaultConfigurationIsVisible = 0
        self.defaultConfigurationName = settings and settings[0].name or ""

# Group/Files
class PBXFileReference(XCodeNode):
    def __init__(self, name, path, filetype = '', sourcetree = "SOURCE_ROOT"):
        XCodeNode.__init__(self)
        self.fileEncoding = 4
        if not filetype:
            _, ext = os.path.splitext(name)
            filetype = MAP_EXT.get(ext, 'text')
        self.lastKnownFileType = filetype
        self.name = name
        self.path = path
        self.sourceTree = sourcetree

class PBXGroup(XCodeNode):
    def __init__(self, name, sourcetree = "<group>"):
        XCodeNode.__init__(self)
        self.children = []
        self.name = name
        self.sourceTree = sourcetree

    def add(self, root, sources):
        folders = {}
        def folder(n):
            if not n.is_child_of(root):
                return self
            try:
                return folders[n]
            except KeyError:
                f = PBXGroup(n.name)
                p = folder(n.parent)
                folders[n] = f
                p.children.append(f)
                return f
        for s in sources:
            f = folder(s.parent)
            source = PBXFileReference(s.name, s.abspath())
            f.children.append(source)


# Targets
class PBXLegacyTarget(XCodeNode):
    def __init__(self, action, target=''):
        XCodeNode.__init__(self)
        self.buildConfigurationList = XCConfigurationList([XCBuildConfiguration('waf', {})])
        if not target:
            self.buildArgumentsString = "%s %s" % ("waf_xcode.sh")
        else:
            self.buildArgumentsString = "%s --targets=%s" % ("waf_xcode.sh", target)
        self.buildPhases = []
        self.buildToolPath = "/bin/bash" #sys.executable
        self.buildWorkingDirectory = ""
        self.dependencies = []
        self.name = target or action
        self.productName = target or action
        self.passBuildSettingsInEnvironment = 0

class PBXShellScriptBuildPhase(XCodeNode):
    def __init__(self, action, target):
        XCodeNode.__init__(self)
        self.buildActionMask = 2147483647
        self.files = []
        self.inputPaths = []
        self.outputPaths = []
        self.runOnlyForDeploymentPostProcessing = 0
        self.shellPath = "/bin/sh"
        self.shellScript = "%s %s %s --targets=%s" % (sys.executable, sys.argv[0], action, target)

class PBXNativeTarget(XCodeNode):
    def __init__(self, action, target, node):
        XCodeNode.__init__(self)
        conf = XCBuildConfiguration('waf', {'PRODUCT_NAME':target, 'CONFIGURATION_BUILD_DIR':node.parent.abspath()})
        self.buildConfigurationList = XCConfigurationList([conf])
        self.buildPhases = [PBXShellScriptBuildPhase(action, target)]
        self.buildRules = []
        self.dependencies = []
        self.name = target
        self.productName = target
        self.productType = "com.apple.product-type.application"
        self.productReference = PBXFileReference(target, node.abspath(), 'wrapper.application', 'BUILT_PRODUCTS_DIR')

# Root project object
class PBXProject(XCodeNode):
    def __init__(self, name, version):
        XCodeNode.__init__(self)
        self.buildConfigurationList = XCConfigurationList([XCBuildConfiguration('waf', {})])
        self.compatibilityVersion = version[0]
        self.hasScannedForEncodings = 1;
        self.mainGroup = PBXGroup(name)
        self.projectRoot = ""
        self.projectDirPath = ""
        self.targets = []
        self._objectVersion = version[1]
        self._output = PBXGroup('out')
        self.mainGroup.children.append(self._output)

    def write(self, file):
        w = file.write
        w("// !$*UTF8*$!\n")
        w("{\n")
        w("\tarchiveVersion = 1;\n")
        w("\tclasses = {\n")
        w("\t};\n")
        w("\tobjectVersion = %d;\n" % self._objectVersion)
        w("\tobjects = {\n\n")

        XCodeNode.write(self, file)

        w("\t};\n")
        w("\trootObject = %s;\n" % self._id)
        w("}\n")

    def add_task_gen(self, tg):
        if not getattr(tg, 'mac_app', False):
            self.targets.append(PBXLegacyTarget('build', tg.name))
        else:
            target = PBXNativeTarget('build', tg.name, tg.link_task.outputs[0].change_ext('.app'))
            self.targets.append(target)
            self._output.children.append(target.productReference)


def create_shell_script():
        cwd = os.getcwd()
        waf_bin = cwd + '/waf'
        src_dir = cwd + '/src'
        f = open("waf_xcode.sh", 'w+')
        f.write('(' +  waf_bin + ' $1 ) 2> >( sed -E "s|../src/([^/][a-zA-Z/_]+\\.cpp)|' + src_dir + '/\\1|g;s|../src/([^/][a-zA-Z/_]+\\.hpp)|' + src_dir + '/\\1|g" >&2 )')
        f.close()

class xcode(Build.BuildContext):
    cmd = 'xcode'
    fun = 'build'
    create_shell_script()

    def collect_source(self, tg):
        source_files = tg.to_nodes(getattr(tg, 'source', []))
        plist_files = tg.to_nodes(getattr(tg, 'mac_plist', []))
        resource_files = [tg.path.find_node(i) for i in Utils.to_list(getattr(tg, 'mac_resources', []))]
        include_dirs = Utils.to_list(getattr(tg, 'includes', [])) + Utils.to_list(getattr(tg, 'export_dirs', []))
        include_files = []
        for x in include_dirs:
            if not isinstance(x, str):
                include_files.append(x)
                continue
            d = tg.path.find_node(x)
            if d:
                lst = [y for y in d.ant_glob(HEADERS_GLOB, flat=False)]
                include_files.extend(lst)

        # remove duplicates
        source = list(set(source_files + plist_files + resource_files + include_files))
        source.sort(key=lambda x: x.abspath())
        return source

    def execute(self):
        """
        Entry point
        """
        self.restore()
        if not self.all_envs:
            self.load_envs()
        self.recurse([self.run_dir])

        appname = getattr(Context.g_module, Context.APPNAME, os.path.basename(self.srcnode.abspath()))
        p = PBXProject(appname, ('Xcode 3.2', 46))

        for g in self.groups:
            for tg in g:
                if not isinstance(tg, TaskGen.task_gen):
                    continue

                tg.post()

                features = Utils.to_list(getattr(tg, 'features', ''))

                group = PBXGroup(tg.name)
                group.add(tg.path, self.collect_source(tg))
                p.mainGroup.children.append(group)

                if 'cprogram' or 'cxxprogram' in features:
                    p.add_task_gen(tg)



        # targets that don't produce the executable but that you might want to run
        p.targets.append(PBXLegacyTarget('configure'))
        p.targets.append(PBXLegacyTarget('dist'))
        p.targets.append(PBXLegacyTarget('install'))
        p.targets.append(PBXLegacyTarget('check'))
        node = self.srcnode.make_node('%s.xcodeproj' % appname)
        node.mkdir()
        node = node.make_node('project.pbxproj')
        p.write(open(node.abspath(), 'w'))
