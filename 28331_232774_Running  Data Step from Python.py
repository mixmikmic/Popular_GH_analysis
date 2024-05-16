import swat

conn = swat.CAS(host, port, username, password)

cls = conn.read_csv('https://raw.githubusercontent.com/sassoftware/sas-viya-programming/master/data/class.csv',
                    casout=dict(name='class', caslib='casuser'))
cls

out = conn.datastep.runcode('''
   data bmi(caslib='casuser');
      set class(caslib='casuser');
      BMI = weight / (height**2) * 703;
   run;
''')
out

bmi = out.OutputCasTables.ix[0, 'casTable']
bmi.to_frame()

bmi2 = cls.datastep('''BMI = weight / (height**2) * 703''')
bmi2.to_frame()

get_ipython().magic('load_ext swat.cas.magics')

get_ipython().run_cell_magic('casds', '--output out2 conn', "\ndata bmi3(caslib='casuser');\n   set class(caslib='casuser');\n   BMI = weight / (height**2) * 703;\nrun;")

bmi3 = out2.OutputCasTables.ix[0, 'casTable']
bmi3.to_frame()

conn.close()



