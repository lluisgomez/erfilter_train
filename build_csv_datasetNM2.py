import os
import subprocess

f = open('char_datasetNM2.csv', 'w')

#obtain positive samples 
traindbdir = "./data/char/"

for filename in os.listdir(traindbdir):
  print("processing "+filename);
	
  out = subprocess.check_output(["./extract_featuresNM2",traindbdir+filename])
	
  if ("Non-integer" in out):
		print "ERROR: Non-integer Euler number"
	
  else:
		if (out != ''):
			out = out.replace("\n","\nC,",out.count("\n")-1)
			f.write("C,"+out)

#obtain negative samples 
traindbdir = "./data/nochar/"

for filename in os.listdir(traindbdir):
  print("processing "+filename);
	
  out = subprocess.check_output(["./extract_featuresNM2",traindbdir+filename])
	
  if ("Non-integer" in out):
		print "ERROR: Non-integer Euler number"
	
  else:
		if (out != ''):
			out = out.replace("\n","\nN,",out.count("\n")-1)
			f.write("N,"+out)

f.close()
