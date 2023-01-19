```python

######################################
## Convert indexes to datetime type ##
######################################

from dateutil import parser
from datetime import datetime

data2 = [i for i in data1]
for i in range(86):
  old_index = [str(x) for x in data2[i].index.tolist()]
  new_index = []
  for j in old_index:
    date_time = parser.parse(j)
    new_index.append(date_time)
    
  data2[i] = data2[i].rename(index=dict(zip(old_index,new_index)))
  
###################################
## Identify non-cinsecutive time ##
###################################

def consecutive(i):
  list1 = data2[i].index.tolist()
  k = 1
  subsequ = [0]
  res = 1
  while((res==1)&(k<len(list1))):
    diff = list1[k]-list1[k-1]
    res = int(divmod(diff.total_seconds(), 60)[0])
    if(res!=1):
      subsequ.append(k) 
    k+=1
    
  subsequ.append(len(list1)-1)

  return subsequ
  
#####################
## Data Labelling ##
#####################

def labelling(dat):
    hr, ma = 0, 0
    for j in range(30):
        if (dat["HR.ts_mean"][j]>100):
            hr+=1
        if (dat["MAP.ts_mean"][j]<60):
            ma+=1
    phr = (hr*100)/30
    pma = (ma*100)/30
    deci = [0,0]
    if (pma>=90): deci[0] = 1
    if (phr>=90): deci[1] = 1

    return deci
    
######################
## Compile the data ##
######################

def data_compiled():
    result = []
    for i in range(86):
      nOBi = data2[i].shape[0]
      data2[i]["target_HR"] = [0]*nOBi
      data2[i]["target_MAP"] = [0]*nOBi
      tmp = pd.DataFrame(columns = data2[3].columns)
      subdat_i = consecutive(i)
      n = len(subdat_i)
      for j in range(1,n):
        inter = subdat_i[j]-subdat_i[j-1]
        if(inter>150):
          data_d = data2[i].iloc[subdat_i[j-1]:subdat_i[j-1]+60,:]
          deci = labelling(data2[i].iloc[subdat_i[j-1]+120:subdat_i[j-1]+150,:])
          if deci[0] == 1:
            data_d["target_MAP"] = [1]*60
          if deci[1] == 1:
            data_d["target_HR"] = [1]*60
          m = subdat_i[j-1]+30
          while(m+150<subdat_i[j]):
            data_d1 =  data2[i].iloc[m:m+60,:]
            deci = labelling(data2[i].iloc[m+120:m+150,:])
            if(deci[0] == 1):
              data_d1["target_MAP"] = [1]*60
            if(deci[1] == 1):
              data_d1["target_HR"] = [1]*60
            data_d = pd.concat([data_d, data_d1], ignore_index = True)
            m = m+30
          tmp = pd.concat([tmp, data_d], ignore_index = True)

      result.append(tmp)

    return result
    
final_data = data_compiled()

```
