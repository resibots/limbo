import sys, os, stat
import subprocess
import commands

json_ok = True
try:
   import simplejson
except:
   json_ok = False
   print "WARNING simplejson not found some function may not work"

import glob

def options(opt):
	opt.add_option('--qsub', type='string', help='json file to submit to torque', dest='qsub')

def create_variants(bld, source, uselib_local,
                    uselib, variants, includes=". ../", 
                    cxxflags='',
                    json='',
                    target=''):
   # the basic one
   #   tgt = bld.new_task_gen('cxx', 'program')
   #   tgt.source = source
   #   tgt.includes = includes
   #   tgt.uselib_local = uselib_local
   #   tgt.uselib = uselib
   # the variants
   # we create a basic file
   if not target: 
      tmp = source.replace('.cpp', '')
   else:
      tmp=target
   tgt = bld.program(features = 'cxx',
                     source = source,
                     target = tmp,
                     includes = includes,
                     uselib = uselib,
                     use = uselib_local)
   c_src = bld.path.abspath() + '/'
   for v in variants:
      # create file
      suff = ''
      for d in v.split(' '): suff += d.lower() + '_'
      src_fname = tmp + '_' + suff[0:len(suff) - 1] + '.cpp'
      bin_fname = tmp + '_' + suff[0:len(suff) - 1]
      f = open(c_src + src_fname, 'w')
      f.write("// THIS IS A GENERATED FILE - DO NOT EDIT\n")
      for d in v.split(' '): f.write("#define " + d + "\n")
      f.write("#line 1 \"" + c_src + source + "\"\n")
      code = open(c_src + source, 'r')
      for line in code: f.write(line)
      bin_name = src_fname.replace('.cpp', '')
      bin_name = os.path.basename(bin_name)
      # create build
      tgt = bld.program(features = 'cxx',
                        source = src_fname,
                        target = bin_fname,
                        includes = includes,
                        uselib = uselib,
                        use = uselib_local)

def _sub_script(tpl, conf_file):
   if os.environ.has_key('LD_LIBRARY_PATH'):
      ld_lib_path = os.environ['LD_LIBRARY_PATH']
   else:
      ld_lib_path = "''"
   home = os.environ['HOME']
   print 'LD_LIBRARY_PATH=' + ld_lib_path
    # parse conf
   conf = simplejson.load(open(conf_file))
   exps = conf['exps']
   nb_runs = conf['nb_runs']
   res_dir = conf['res_dir']
   bin_dir = conf['bin_dir']
   wall_time = conf['wall_time']
   use_mpi = "false"
   try: use_mpi = conf['use_mpi']
   except: use_mpi = "false"
   try: nb_cores = conf['nb_cores']
   except: nb_cores = 1
   try: args = conf['args']
   except: args = ''
   email = conf['email']
   if (use_mpi == "true"): 
      ppn = '1'
      mpirun = 'mpirun'
   else:
#      nb_cores = 1; 
      ppn = "8"
      mpirun = ''

   fnames = []
   for i in range(0, nb_runs):
      for e in exps:
         directory = res_dir + "/" + e + "/exp_" + str(i) 
         try:
            os.makedirs(directory)
         except:
            print "WARNING, dir:" + directory + " not be created"
         subprocess.call('cp ' + bin_dir + '/' + e + ' ' + directory, shell=True)
         fname = directory + "/" + e + "_" + str(i) + ".job"
         f = open(fname, "w")
         f.write(tpl
                 .replace("@exp", e)
                 .replace("@email", email)
                 .replace("@ld_lib_path", ld_lib_path)
                 .replace("@wall_time", wall_time)
                 .replace("@dir", directory)
                 .replace("@nb_cores", str(nb_cores))
                 .replace("@ppn", ppn)
                 .replace("@exec", mpirun + ' ' + directory + '/' + e + ' ' + args))
         f.close()
         os.chmod(fname, stat.S_IEXEC | stat.S_IREAD | stat.S_IWRITE)
         fnames += [(fname, directory)]
   return fnames

def qsub(conf_file):
   tpl = """#!/bin/sh
#? nom du job affiche
#PBS -N @exp
#PBS -o stdout
#PBS -b stderr
#PBS -M @email
# maximum execution time
#PBS -l walltime=@wall_time
# mail parameters
#PBS -m abe
# number of nodes
#PBS -l nodes=@nb_cores:ppn=@ppn
#PBS -l pmem=5200mb -l mem=5200mb
export LD_LIBRARY_PATH=@ld_lib_path
exec @exec
"""
   fnames = _sub_script(tpl, conf_file)
   for (fname, directory) in fnames:
      s = "qsub -d " + directory + " " + fname
      print "executing:" + s
      retcode = subprocess.call(s, shell=True, env=None)
      print "qsub returned:" + str(retcode)



def oar(conf_file):
   tpl = """#!/bin/bash
#OAR -l /core=@nb_cores/nodes=1,walltime=@wall_time
#OAR -n @exp
#OAR -O stdout.%jobid%.log
#OAR -E stderr.%jobid%.log
export LD_LIBRARY_PATH=@ld_lib_path
exec @exec
"""
   print 'WARNING [oar]: MPI not supported yet'
   fnames = _sub_script(tpl, conf_file)
   for (fname,directory) in fnames:
      s = "oarsub -d " + directory + " -S " + fname
      print "executing:" + s
      retcode = subprocess.call(s, shell=True, env=None)
      print "oarsub returned:" + str(retcode)
