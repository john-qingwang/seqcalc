# seqcalc
A tool for intact mass spec data sequence identification.

To use this tool we need the following:
   * sequence_filename: the full path of the file that contains the protein sequences. an example is given in fake_sequence.txt
   * mass_lib_filename: the full path to a text file that specifies the one to one match of a molecule name to its mass
   * output_filename: the full path to a file where the result is dumpped into. Defalut is "output.txt" in the working directory
   * tol: the absolute tolerance of the sequence searching
   * target_mass: the total mass that the clipped sequence(s) will have

A sample command to run this tool is:

```py sequence_detector.py sequence_filename=51671.txt mass_lib_filename=aa_mass.txt tol=10.0 target_mass=3368```
