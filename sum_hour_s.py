import pandas as pd

# Excelファイルの読み込み
df = pd.read_excel('/Users/user_name/Documents/GitHub/excel_io/input_file.xlsx', sheet_name='Sheet1')

# 年月日列を datetime 形式に変換
df['年月日'] = pd.to_datetime(df['年月日'], format='%Y/%m/%d')

# ユーザーに科目、年度、および所属を入力してもらう
subject = input("出力したい科目を入力してください（例：国語、数学、英語など）: ")
year = input("出力したい年度を入力してください（例：2024）: ")
affiliation = input("出力したい所属を入力してください（例：D、Aなど）: ")

# 入力チェック
if not year.isdigit():
    print("無効な年度が入力されました。数値を入力してください。")
    exit()

year = int(year)

if subject not in df['科目'].unique() or affiliation not in df['所属'].unique():
    print(f"指定した科目「{subject}」、年度「{year}」、または所属「{affiliation}」はデータに存在しません。")
    exit()

# フィルタリング
df_filtered = df[(df['所属'] == affiliation) & (df['科目'] == subject)].copy()

# 年度と月を抽出
df_filtered['年度'] = df_filtered['年月日'].dt.year
df_filtered['月'] = df_filtered['年月日'].dt.month

# 会計年度の月を計算
df_filtered['会計月'] = df_filtered['月'].apply(lambda x: x if 4 <= x <= 12 else x + 12)

# 年度でフィルタリング
df_filtered = df_filtered[((df_filtered['年度'] == year) & (df_filtered['月'] >= 4)) | ((df_filtered['年度'] == year + 1) & (df_filtered['月'] <= 3))]

if df_filtered.empty:
    print(f"指定した科目「{subject}」および年度「{year}」に対応するデータは存在しません。")
    exit()

# 集計
result_df = df_filtered.groupby(['氏名', '会計月'])['得点'].sum().unstack(fill_value=0)

# 月の並び順を調整
month_order = [f'{i}月' for i in range(4, 13)] + [f'{i}月' for i in range(1, 4)]
result_df = result_df[month_order]

# 合計行を追加
result_df.loc['合計'] = result_df.sum()

# 科目、年度、所属を追加
result_df['科目'] = subject
result_df['年度'] = year
result_df['所属'] = affiliation

# Excel書き出し
output_file = f'output_{subject}_{year}_{affiliation}.xlsx'
result_df.to_excel(output_file)

print(f"{subject}のデータ（年度：{year}、所属：{affiliation}）が{output_file}に書き出されました。")
