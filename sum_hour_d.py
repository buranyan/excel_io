import pandas as pd

# dtype と parse_dates を指定して読み込み
dtype = {
    '所属': 'str',  # 所属列を文字列型
    '氏名': 'str',  # 氏名列を文字列型
    '得点': 'float32',  # 得点列を float32 型（小数点1桁まで対応）
    '科目': 'str'  # 科目列を文字列型
}

df = pd.read_excel('/Users/mukaikazuhiro/Documents/kuronyan-sleep/local_file/input_file.xlsx',
                   usecols=['所属','氏名','年月日','科目','得点'],
                   dtype=dtype,
                   parse_dates=['年月日'])

# ユーザーに科目、年度、および所属を入力してもらう
subject = input("出力したい科目を入力してください（例：国語、数学、英語など）: ")
year = input("出力したい年度を入力してください（例：2024）: ")
affiliation = input("出力したい所属を入力してください（例：D、Aなど）: ")

# 入力した年度が数値かどうかチェック
if not year.isdigit():
    print("無効な年度が入力されました。数値を入力してください。")
else:
    year = int(year)

    # 科目、年度、所属がデータに存在するか確認
    if subject not in df['科目'].values or affiliation not in df['所属'].values:
        print(f"指定した科目「{subject}」、年度「{year}」、または所属「{affiliation}」はデータに存在しません。")
    else:
        # 所属が指定された値、かつ指定された科目のデータをフィルタリングし、コピーを作成
        df_filtered = df[(df['所属'] == affiliation) & (df['科目'] == subject)].copy()

        # 年度と月を抽出
        df_filtered['年度'] = df_filtered['年月日'].dt.year
        df_filtered['月'] = df_filtered['年月日'].dt.month

        # 4月から12月はその年（year）でフィルタリング、1月から3月は翌年（year+1）でフィルタリング
        df_filtered_4_to_12 = df_filtered[(df_filtered['年度'] == year) & (df_filtered['月'] >= 4) & (df_filtered['月'] <= 12)]
        df_filtered_1_to_3 = df_filtered[(df_filtered['年度'] == year + 1) & (df_filtered['月'] >= 1) & (df_filtered['月'] <= 3)]

        # 両方のデータを結合
        df_filtered = pd.concat([df_filtered_4_to_12, df_filtered_1_to_3])

        # データが空の場合
        if df_filtered.empty:
            print(f"指定した科目「{subject}」および年度「{year}」に対応するデータは存在しません。")
        else:
            # 氏名ごとに月別の得点を格納する辞書を作成
            result_data = {}

            # 所属が指定された値、かつ指定科目および年度のデータを処理
            for _, row in df_filtered.iterrows():
                name = row['氏名']
                month = row['月']
                score = row['得点']

                # 4月〜12月はそのまま月と対応、1月〜3月は13月〜15月に変換
                if month >= 4 and month <= 12:
                    fiscal_month = month  # 4月〜12月はそのまま
                else:
                    fiscal_month = month + 12  # 1月〜3月は13月〜15月に変換

                # 氏名ごとに月別得点を保持（キーが存在しない場合は初期化）
                if name not in result_data:
                    result_data[name] = {f"{i}月": 0 for i in range(1, 16)}  # 1月〜15月まで初期化

                # 月ごとに得点を加算（キーの確認を行う）
                month_key = f"{fiscal_month}月"
                result_data[name][month_key] += score  # 該当月に得点を加算

            # 結果をリスト形式に変換
            output_data = []
            monthly_totals = [0] * 12  # 4月〜3月の合計を格納するリスト
            total_score_all = 0  # 年間合計を初期化

            # 氏名ごとのデータを処理
            for name, months_scores in result_data.items():
                # 月別得点をリストにまとめる（4月が一番左、3月が一番右）
                # 4月〜12月（その年）はそのままで、1月〜3月（翌年）は13月〜15月に対応
                monthly_scores = [months_scores[f"{i}月"] for i in range(4, 13)] + [months_scores[f"{i}月"] for i in range(13, 16)]
                
                # 年間合計を計算
                total_score = sum(monthly_scores)
                
                # 月別得点を集計（各月の合計）
                for i in range(12):
                    monthly_totals[i] += monthly_scores[i]
                
                # 年間合計を全体の合計に加算
                total_score_all += total_score
                
                # 氏名、月別得点リスト、年間得点をまとめる
                output_data.append([name] + monthly_scores + [total_score])

            # 月別の合計行を追加
            output_data.append(['合計'] + monthly_totals + [total_score_all])

            # 科目、年度、所属を出力データに追加
            for row in output_data:
                row.append(subject)  # 科目を追加
                row.append(year)     # 年度を追加
                row.append(affiliation)  # 所属を追加

            # 結果をデータフレームに変換
            # 列の並び順を「氏名、4月〜12月、1月〜3月、年間合計、科目、年度、所属」に変更
            columns = ['氏名'] + [f'{i}月' for i in range(4, 13)] + [f'{i}月' for i in range(1, 4)] + ['年間合計', '科目', '年度', '所属']
            result_df = pd.DataFrame(output_data, columns=columns)

            # 新しいExcelファイルに書き出し（インデックスは不要）
            output_file = f'output_{subject}_{year}_{affiliation}.xlsx'
            result_df.to_excel(output_file, index=False)

            # 完了メッセージ
            print(f"{subject}のデータ（年度：{year}、所属：{affiliation}）が{output_file}に書き出されました。")
