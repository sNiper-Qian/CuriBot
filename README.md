---
language:
  - en
pretty_name: ShapeNetCore
tags:
  - 3D shapes
license: other
extra_gated_heading: Acknowledge license to accept the repository
extra_gated_prompt: >-
  To request access to this ShapeNet repo, you will need to provide your **full name** (please provide both your first and last name), the name of your **advisor or the principal investigator (PI)** of your lab (in the PI/Advisor) fields, and the **school or company** that you are affiliated with (the **Affiliation** field).  
  After requesting access to this ShapeNet repo, you will be considered for access approval. 


  After access approval, you (the "Researcher") receive permission to use the ShapeNet database (the "Database") at Princeton University and Stanford University. In exchange for being able to join the ShapeNet community and receive such permission, Researcher hereby agrees to the following terms and conditions:
  Researcher shall use the Database only for non-commercial research and educational purposes.
  Princeton University and Stanford University make no representations or warranties regarding the Database, including but not limited to warranties of non-infringement or fitness for a particular purpose.
  Researcher accepts full responsibility for his or her use of the Database and shall defend and indemnify Princeton University and Stanford University, including their employees, Trustees, officers and agents, against any and all claims arising from Researcher's use of the Database, including but not limited to Researcher's use of any copies of copyrighted 3D models that he or she may create from the Database.
  Researcher may provide research associates and colleagues with access to the Database provided that they first agree to be bound by these terms and conditions.
  Princeton University and Stanford University reserve the right to terminate Researcher's access to the Database at any time.
  If Researcher is employed by a for-profit, commercial entity, Researcher's employer shall also be bound by these terms and conditions, and Researcher hereby represents that he or she is fully authorized to enter into this agreement on behalf of such employer.
  The law of the State of New Jersey shall apply to all disputes under this agreement.


  For access to the data, please fill in your **full name** (both first and last name), the name of your **advisor or principal investigator (PI)**, and the name of the **school or company** you are affliated with.  
  Please actually fill out the fields (DO NOT put the word "Advisor" for PI/Advisor and the word "School" for "Affiliation", please specify the name of your advisor and the name of your school).
extra_gated_fields:
 Name: text
 PI/Advisor: text
 Affiliation: text
 Purpose: text 
 Country: text
 I agree to use this dataset for non-commercial use ONLY: checkbox
---


This repository contains ShapeNetCore (v2), a subset of [ShapeNet](https://shapenet.org).  
ShapeNetCore is a densely annotated subset of ShapeNet covering 55 common object categories with ~51,300 unique 3D models.  Each model in ShapeNetCore are linked to an appropriate synset in [WordNet 3.0](https://wordnet.princeton.edu/).  

Please see [DATA.md](DATA.md) for details about the data.

If you use ShapeNet data, you agree to abide by the [ShapeNet terms of use](https://shapenet.org/terms). You are only allowed to redistribute the data to your research associates and colleagues provided that they first agree to be bound by these terms and conditions.

If you use this data, please cite the main ShapeNet technical report.
```
@techreport{shapenet2015,
  title       = {{ShapeNet: An Information-Rich 3D Model Repository}},
  author      = {Chang, Angel X. and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and Xiao, Jianxiong and Yi, Li and Yu, Fisher},
  number      = {arXiv:1512.03012 [cs.GR]},
  institution = {Stanford University --- Princeton University --- Toyota Technological Institute at Chicago},
  year        = {2015}
}
```

For more information, please contact us at shapenetwebmaster@gmail.com and indicate ShapeNetCore v2 in the title of your email.
