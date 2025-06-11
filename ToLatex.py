import pandas as pd, os
df = pd.read_csv("figures/4_4_1/summary_raw.csv")

# для компактности выберем нужные поля и округлим
tab = (df[["tag","accuracy","recall","precision","value"]]
       .sort_values("tag")
       .round(3))

# печать в консоль
#print(tab.to_latex(index=False))

# или сразу сохранить
out_path = "figures/4_4_1/summary_table.tex"
with open(out_path, "w") as f:
    f.write(tab.to_latex(index=False,
                         caption="Итоговые метрики RNN по 45~комбинациям",
                         label="tab:knapsack_metrics",
                         column_format="lccccc",
                         escape=False))
print("Файл LaTeX-таблицы сохранён:", out_path)
