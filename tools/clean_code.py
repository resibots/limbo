from subprocess import call
import fnmatch
import os
import sys

# config_data = open('.clang-format', 'r').read()
# config_data = config_data[4:]
# config_data = config_data[:len(config_data)-5]
# config_list = config_data.split('\n')

# config_str = '{'
# for line in config_list:
# 	if line[0] != '#':
# 		config_str = config_str+' '+line+','

# config_str = config_str[:len(config_str)-1] + ' }'

# Format directories and all subfolders using clang-format
def format_dir(folder, extensions):
	matches = []
	for root, dirnames, filenames in os.walk(folder):
		for ext in extensions:
			for filename in fnmatch.filter(filenames, '*'+ext):
				matches.append(os.path.join(root, filename))

	for filename in matches:
		call(["clang-format-3.6", "-i", filename])

# possible extensions
exts = ['.h', '.c', '.hpp', '.cpp', '.hh', '.cc']

if __name__ == "__main__":
	if len(sys.argv) < 1:
		print "Usage: clean_code.py folder1 [folder2] ..."

	# load config file
	curr_dir = os.getcwd()

	# change if needed to find the config file
	if curr_dir[len(curr_dir)-5:] == 'limbo':
		os.chdir('tools')
		call(["cp", ".clang-format", curr_dir])
		# change back to old dir
		os.chdir(curr_dir)

	for arg in sys.argv:
		format_dir(arg, exts)

	if curr_dir[len(curr_dir)-5:] == 'limbo':
		call(["rm", ".clang-format"])
