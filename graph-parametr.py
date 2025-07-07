import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from newspaper import Article
import networkx as nx
import plotly.graph_objects as go
import io
from streamlit_plotly_events import plotly_events

@st.cache_data(show_spinner=False)
def extract_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except:
        return ""

@st.cache_data(show_spinner=True)
def compute_embeddings(texts):
    model = SentenceTransformer('sentence-transformers/LaBSE')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

@st.cache_data(show_spinner=True)
def build_similarity_matrix(embeddings):
    return cosine_similarity(embeddings)

def build_graph(df, sim_matrix, threshold, show_gray):
    G = nx.Graph()
    export_data = []

    for i, row in df.iterrows():
        connected = [
            sim_matrix[i][j]
            for j in range(len(df))
            if i != j and sim_matrix[i][j] >= threshold
        ]
        if show_gray or connected:
            G.add_node(i, label=row['H1-1'], title=row['Address'])

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            sim = sim_matrix[i][j]
            if sim >= threshold:
                G.add_edge(i, j, weight=sim)
                export_data.append({
                    'Source_URL': df.iloc[i]['Address'],
                    'Target_URL': df.iloc[j]['Address'],
                    'Similarity': sim
                })
    return G, export_data

def create_plot(G):
    pos = nx.spring_layout(G, seed=42, k=0.5)
    edge_x, edge_y = [], []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, labels, hover = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(G.nodes[node]['label'])
        hover.append(G.nodes[node]['title'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=labels,
        hovertext=hover,
        hoverinfo='text',
        textposition="top center",
        marker=dict(
            color='MediumPurple',
            size=15,
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title_text="",
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig

def main():
    st.set_page_config(page_title="Визуализация релевантности статей", layout="wide")
    st.title("🔗 Визуализация релевантности текстов статей с помощью BERT")

    uploaded_file = st.file_uploader("Загрузите CSV с колонками 'Address' и 'H1-1'", type=["csv"])
    if not uploaded_file:
        return

    df = pd.read_csv(uploaded_file)
    if 'Address' not in df.columns or 'H1-1' not in df.columns:
        st.error("CSV должен содержать колонки 'Address' и 'H1-1'")
        return

    df = df.dropna(subset=['Address', 'H1-1']).reset_index(drop=True)
    urls = df['Address'].astype(str).tolist()

    if 'texts' not in st.session_state:
        texts = []
        progress_bar = st.progress(0)
        status = st.empty()
        for i, url in enumerate(urls):
            texts.append(extract_article_text(url))
            progress_bar.progress((i + 1) / len(urls))
            status.text(f"Обработка {i + 1} из {len(urls)}")
        st.session_state.texts = texts
    else:
        texts = st.session_state.texts

    df['text'] = texts
    df = df[df['text'].str.strip() != ''].reset_index(drop=True)

    if df.empty:
        st.warning("Не удалось извлечь тексты ни с одного URL")
        return

    st.success(f"Успешно обработано {len(df)} статей")

    if 'embeddings' not in st.session_state:
        embeddings = compute_embeddings(df['text'].tolist())
        st.session_state.embeddings = embeddings
    else:
        embeddings = st.session_state.embeddings

    if 'sim_matrix' not in st.session_state:
        sim_matrix = build_similarity_matrix(embeddings)
        st.session_state.sim_matrix = sim_matrix
    else:
        sim_matrix = st.session_state.sim_matrix

    threshold = st.slider("Порог релевантности", 0.3, 1.0, 0.7, step=0.01)
    show_gray = st.checkbox("Показывать одиночные узлы (без связей)", True)

    need_rebuild = (
        'graph' not in st.session_state
        or 'export_data' not in st.session_state
        or st.session_state.get('last_threshold') != threshold
        or st.session_state.get('last_show_gray') != show_gray
    )

    if need_rebuild:
        G, export_data = build_graph(df, st.session_state.sim_matrix, threshold, show_gray)
        st.session_state.graph = G
        st.session_state.export_data = export_data
        st.session_state.last_threshold = threshold
        st.session_state.last_show_gray = show_gray
    else:
        G = st.session_state.graph
        export_data = st.session_state.export_data

    fig = create_plot(G)

    st.subheader("Граф релевантности")
    selected_points = plotly_events(fig, click_event=True, hover_event=False, key="plotly_graph")

    if selected_points:
        index = selected_points[0]['pointIndex']
        node_ids = list(G.nodes)
        if index < len(node_ids):
            node_id = node_ids[index]
            clicked_url = G.nodes[node_id]['title']

            neighbors = list(G.neighbors(node_id))
            data = []
            for n in neighbors:
                url = G.nodes[n]['title']
                weight = G.get_edge_data(node_id, n)['weight']
                data.append((url, weight))

            st.markdown(f"**Выбранная статья:** [{clicked_url}]({clicked_url})")

            if data:
                table_html = """
                <table>
                  <thead>
                    <tr>
                      <th>Связанная страница</th>
                      <th>Релевантность</th>
                    </tr>
                  </thead>
                  <tbody>
                """
                for url, weight in data:
                    table_html += f'<tr><td><a href="{url}" target="_blank">{url}</a></td><td>{weight:.2f}</td></tr>'
                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.write("Связанных страниц нет.")
        else:
            st.error("Ошибка: выбрана некорректная точка.")

    if export_data:
        csv_buffer = io.StringIO()
        pd.DataFrame(export_data).to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            "📥 Скачать CSV с релевантными парами",
            csv_buffer.getvalue(),
            file_name="relevant_pages.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
