# 専門家AIアシスタント
# (Streamlitアプリケーション)
#
# このモジュールは、LangChainとOpenAIのGPT-4o-miniモデルを使用して、
# 様々な専門家の役割を演じるAIチャットボットを提供します。
#
# 主な機能:
#     - 5種類の専門家タイプから選択可能
#     - ユーザーからのリクエストに基づいて専門的な回答を生成
#     - Streamlitを使用した対話的なWebインターフェース
# Functions:
#     get_answer(specialist_type: str, user_request: str) -> str:
#         指定された専門家タイプとユーザーリクエストに基づいて、
#         LLMから回答を取得します。
#         Args:
#             specialist_type (str): 専門家のタイプ（役割）
#             user_request (str): ユーザーからのリクエスト内容
#         Returns:
#             str: LLMが生成した回答テキスト
# 使用例:
#     アプリケーションを起動:
#     ```
#     streamlit run app.py
#     ```
# 必要な環境変数:
#     - OPENAI_API_KEY: OpenAI APIキー（.envファイルに設定）
# 専門家タイプ:
#     1. ビジネス戦略、プロセス改善の専門家
#     2. キャンペーン設計、ターゲット分析の専門家
#     3. 財務分析、投資戦略の専門家
#     4. スケジュール管理、リスク評価の専門家
#     5. 採用戦略、組織開発の専門家


# モジュール取込
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# 環境変数読込
load_dotenv()

# インスタンス生成(LLM)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True, max_tokens=1000)

# 変数初期化
messages = []   # 会話履歴リスト

# 関数：get_answer
# 専門家タイプとユーザーリクエストを受け取り、回答を返す
def get_answer(specialist_type, user_request):

    # グローバル変数の会話履歴リストを使用
    global messages

    system_role = f"あなたは{specialist_type}です。"
    system_role += f"\n専門的な知識を活かして、ユーザーのリクエストに対して具体的かつ実践的なアドバイスを提供してください。"
    system_role += f"\n専門分野ではないリクエストには他の専門家に相談するように促してください。"
    system_role += f"\n回答は簡潔に、わかりやすく説明してください。"

    # 会話リストに役割とリクエストを追加
    system_message_content = system_role # 役割
    human_message_content = user_request # リクエスト
    messages.append(SystemMessage(content = system_message_content))
    messages.append(HumanMessage(content = human_message_content))

    # LLMのレスポンスを取得
    result = llm.invoke(messages)

    # LLMの回答部分のみ返却
    return result.content

# モジュール取込
import streamlit as st

# タイトル表示
st.title("専門家AIアシスタント")

# テキストを入力するUI
text_input_value = st.text_input(label="リクエストを入力してください。")

# 専門家タイプの選択肢リスト
radio_1 = [
    "ビジネス戦略、プロセス改善の専門家",
    "キャンペーン設計、ターゲット分析の専門家",
    "財務分析、投資戦略の専門家",
    "スケジュール管理、リスク評価の専門家",
    "採用戦略、組織開発の専門家"
]

# 専門家の種類を選択するUI
selected_item = st.radio(
    "LLMに振る舞わせる専門家の種類を選択してください。",
    [
        radio_1[0],
        radio_1[1],
        radio_1[2],
        radio_1[3],
        radio_1[4],
    ]
)

if st.button("実行"):
    st.divider()

    if text_input_value.strip() == "":
        st.write("リクエストが入力されていません。")
    else:
        st.write("### 回答:")
        st.write(get_answer(specialist_type=selected_item, user_request=text_input_value))