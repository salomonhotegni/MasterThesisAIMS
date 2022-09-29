```python

################################################
## Function to compute the Evaluation Metrics ##
################################################

def EvaMec(y_test, y_predict):
  n = len(y_test)
  numEP = int(n/60)
  TEI = 0
  NormAct = 0
  CorPosPredEvent = 0
  FalPosPredEvent = 0
  FalAla = []
  aveFA = 0
  verif2 = 0
  RP = "nan"
  ER = "nan"
  aveAT = "nan"
  aveAT = "nan"

  for i in range(numEP):
    alarm = y_test[i*60]
    verif1 = []
    if alarm == 1:
      TEI += 1
      check = 0
      for j in range(i*60, (i+1)*60):
        if y_predict[j] == 1:
          verif1.append((i+1)*60-j)
          check += 1
      if check != 0:
        CorPosPredEvent += 1
        verif2 += verif1[0]
    if alarm == 0:
      NormAct += 1
      check = 0
      for j in range(i*60, (i+1)*60):
        if y_predict[j] == 1:
          check += 1
      if check != 0:
        FalPosPredEvent += 1
        FalAla.append(check)

  IPP = FalPosPredEvent
  CPP = CorPosPredEvent
  if (CPP+IPP)!=0:
    RP = CPP/(CPP+IPP)
  if TEI != 0:
    ER = CPP/TEI
  if NormAct != 0:
    aveFA = sum(FalAla)/NormAct
  if CPP != 0:
    aveAT = verif2/CPP
  

  eva_met = pd.DataFrame({"TEI": TEI, "IPP":IPP, "CPP": CPP,
                          "ER": ER, "RP": RP, "aveFA": aveFA, "aveAT": aveAT}, index = ["EM"])
  
  return eva_met

```
