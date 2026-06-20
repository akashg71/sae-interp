# Feature catalogue (Phase 1)

Model `gpt2` · hook `blocks.8.hook_resid_pre` · SAE `gpt2-small-res-jb`

Fill in the **Label** column after reading the examples.

| Feature | Density | Logit-lens hint | Top example | Label (you fill in) |
|--------:|--------:|-----------------|-------------|---------------------|
| 7137 | 3.24% | SourceFile, Interstitial, unte, rote, otions | `act= 19.41 \|      <p1></【p】1>\n     ` | code — XML tags / config file syntax |
| 13481 | 1.47% |  usually, usually, often, pmwiki,  invariably | `act=  6.85 \|  lifetime of a patient, it may be【 necessary】 to perform a joint replacement procedure on the` | frequency/habit language ("usually", "sometimes", "depending on" |
| 16836 | 1.12% |  resorted,  decided,  devised,  resort,  opted | `act= 17.73 \|  and I can't find the problem,【 so】 I was hoping you guys could help me` | pivot/consequence words ("so", "therefore", "thus", "instead") |
| 488 | 1.07% |  Miscellaneous,  Ibid,  Conclusion,  Paras,  Utilities | `act= 16.67 \| ').QuerystringParser,\n  【 】 OctetParser     ` | code/markdown structure — imports, headers, indentation |
| 9577 | 1.37% |  inmates,  patients,  prisoners,  detainees,  captives | `act=  9.85 \|  Apollo Hospital in Delhi have saved the life【 of】 a 14-year-old Iraqi student` | Institutionally confined or vulnerable people — patients, prisoners, detainees, addicts, juveniles under care/custody |
