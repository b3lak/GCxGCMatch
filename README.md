# GCxGCMatch
This repository was made by Caleb Marx. The programs included are tools used for comprehensive multidimensional chromatography data analysis. 

AIModel2:
ChangeColumnNames: converts two column labels from <sup>1</sup>t<sub>R</sub>	<sup>2</sup>t<sub>R</sub> to RT1 and RT2 which later files requires to function.
ColumnCheck: created as a simple check to ensure that the column names are all correct before the non-targeted search 
CombineFiles: Combines excel files together
ConvertToRI: Requires the location (primary and secondary retention times) of the alkanes and the following PAHs (Toluene, Naphthalene, Chrysene, and Benzo a Pyrene
Excel Splitter: simple UI to split sheets from a master file into sub files
RINonTargetList:
RINonTargetSearch:



The workflow created to use these files can be found below:
1	Integrate all samples
2	Fix Column Names
3	Ensure second retention times are correctly formatted
4	Combine all samples to create Master List
5	Convert To RI
6 Check Column Names
7	Create Master Search List
8	Separate Master List into sub groups (training, test: 87d 87e etc…)
9	Apply the search List to each subgroup
10	Reformat Area Results to include a classification and a binary compound presence compound
11	Train model
12	Test unknown sample
