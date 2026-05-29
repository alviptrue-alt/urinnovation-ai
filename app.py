# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

# ============================================================================
# 1. НАСТРОЙКА СТРАНИЦЫ
# ============================================================================
st.set_page_config(
    page_title="ЮрИнновация",
    page_icon="⚖️",  # можно эмодзи
    layout="wide"
)

# ========== СВЕТЛАЯ ЛАВАНДОВАЯ ТЕМА ==========
st.markdown("""
<style>
    /* Основной фон приложения */
    .stApp {
        background: linear-gradient(135deg, #f5f0ff 0%, #ebe4fc 100%);
    }
    
    /* Текст */
    .stApp, .stMarkdown, p, div, span, label {
        color: #2d1b4e !important;
    }
    
    /* Заголовки */
    h1, h2, h3, h4, h5, h6 {
        color: #6a3e9e !important;
        font-weight: 600 !important;
    }
    
    h1 {
        background: linear-gradient(90deg, #7e5a9e, #9b6bb3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Боковая панель */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0eaff 0%, #e5daf5 100%);
        border-right: 1px solid #d4c4f0;
    }
    
    [data-testid="stSidebar"] * {
        color: #3d2b5e !important;
    }
    
    /* Кнопки */
    .stButton > button {
        background: linear-gradient(90deg, #b87ed6, #9b59b6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(107, 70, 147, 0.2) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(155, 89, 182, 0.3) !important;
        background: linear-gradient(90deg, #a86ec2, #8b48a8) !important;
    }
    
    /* Поля ввода */
    .stTextArea textarea, .stTextInput input {
        background-color: #ffffff !important;
        border: 1px solid #d4c4f0 !important;
        border-radius: 12px !important;
        color: #2d1b4e !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #9b59b6 !important;
        box-shadow: 0 0 8px rgba(155, 89, 182, 0.3) !important;
    }
    
    /* Radio-кнопки */
    .stRadio > div {
        background-color: #f5f0ff !important;
        border-radius: 12px !important;
        padding: 10px !important;
        border: 1px solid #d4c4f0 !important;
    }
    
    .stRadio label {
        color: #3d2b5e !important;
    }
    
    /* Метрики */
    [data-testid="stMetricValue"] {
        color: #7e5a9e !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #9b6bb3 !important;
    }
    
    /* Информационные блоки */
    .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 12px !important;
    }
    
    .stInfo {
        background-color: #f0eaff !important;
        border-left: 5px solid #9b59b6 !important;
        color: #2d1b4e !important;
    }
    
    /* Expandader (раскрывающиеся блоки) */
    .streamlit-expanderHeader {
        background-color: #f5f0ff !important;
        border-radius: 12px !important;
        color: #6a3e9e !important;
        border: 1px solid #d4c4f0 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        border-radius: 12px !important;
    }
    
    /* Прогресс-бар */
    .stProgress > div > div {
        background-color: #9b59b6 !important;
    }
    
    /* Кастомный блок результата */
    .result-card {
        background: linear-gradient(135deg, #f0eaff 0%, #e5daf5 100%);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #d4c4f0;
        box-shadow: 0 4px 12px rgba(107, 70, 147, 0.1);
    }
    
    /* Скроллбар */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #e5daf5;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #b87ed6;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #9b59b6;
    }
    
    /* Ссылки */
    a {
        color: #7e5a9e !important;
        text-decoration: none !important;
    }
    
    a:hover {
        color: #9b59b6 !important;
        text-decoration: underline !important;
    }
    
    /* Выпадающие списки */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border-color: #d4c4f0 !important;
    }
    
    /* Сообщения об ошибках и предупреждения */
    .stAlert {
        background-color: #f0eaff !important;
    }
    
    /* Блок с похожими делами */
    .similar-case-card {
        background-color: #f9f5ff;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #9b59b6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def find_column(df, possible_names, default=None):
    """
    Ищет колонку в DataFrame по списку возможных названий.
    
    Args:
        df: pandas DataFrame
        possible_names: список возможных названий колонки
        default: значение по умолчанию, если колонка не найдена
    
    Returns:
        название найденной колонки или default
    """
    for name in possible_names:
        if name in df.columns:
            return name
    return default


def clean_description_column(df, col_name):
    """Очищает колонку с описаниями от пустых значений"""
    if col_name is None:
        return df
    
    nan_count = df[col_name].isna().sum()
    if nan_count > 0:
        st.info(f"📊 Обнаружено {nan_count} пустых описаний. Они будут заменены.")
    
    df[col_name] = df[col_name].fillna('').astype(str)
    df.loc[df[col_name].str.strip() == '', col_name] = "Описание дела отсутствует"
    return df

def get_outcome_color(outcome_text):
    outcome_lower = str(outcome_text).lower()
    # Расширенный список ключевых слов для удовлетворения
    if any(word in outcome_lower for word in ["удовлетвор", "удовлетворен", "иск удовлетворен", "частично удовлетворен"]):
        return "green", "🟢"
    # Расширенный список для отказа
    elif any(word in outcome_lower for word in ["отказа", "отказан", "в иске отказано", "отказать"]):
        return "red", "🔴"
    elif "прекращ" in outcome_lower:
        return "orange", "🟠"
    else:
        return "gray", "⚪"


def get_case_number(row, idx):
    """
    Универсально получает номер дела из разных возможных колонок
    """
    possible_columns = [
        'Номер дела', 'Номер', '№', 'Номер дела', 
        'Case Number', 'case_number', 'ID', 'id', 
        'Номер_дела', 'НомерДела', 'Дело №'
    ]
    
    for col in possible_columns:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip() != '' and str(val).strip() != 'nan':
                return str(val)
    
    # Если ничего не нашли, возвращаем порядковый номер
    return f"Дело {idx}"


# ============================================================================
# 3. ЗАГРУЗКА ДАННЫХ
# ============================================================================

@st.cache_data
def load_cases():
    """Загружает файл с данными и автоматически определяет колонки"""
    
    # Возможные имена файлов
    possible_files = [
        'df_nov29_100pages_1.xlsx',
        'df_nov29_100pages_1.xls',
        'df_nov29_1800pages_1.xlsx',
        'cases_with_texts.xlsx',
        'cases_processed.xlsx',
        'df_nov29_100pages_1.xlsx'
    ]
    
    file_name = None
    for name in possible_files:
        if os.path.exists(name):
            file_name = name
            break
    
    if file_name is None:
        st.error("❌ Файл с данными не найден!")
        st.write("📁 Искала файлы:", possible_files)
        st.write("📄 Файлы в папке:", os.listdir('.'))
        return None
    
    # Загружаем файл
    try:
        df = pd.read_excel(file_name)
        st.success(f"✅ Загружен файл: {file_name}")
    except Exception as e:
        try:
            df = pd.read_csv(file_name, encoding='utf-8')
            st.success(f"✅ Загружен файл (как CSV): {file_name}")
        except Exception as e2:
            st.error(f"❌ Не удалось загрузить файл: {e2}")
            return None
    
    # Автоматически определяем нужные колонки
    description_col = find_column(df, [
        'Описание', 'описание', 'Текст', 'Содержание', 
        'Description', 'description', 'Текст дела'
    ])
    
    outcome_col = find_column(df, [
        'Текущее состояние', 'Текущее_состояние', 'Исход', 
        'Решение', 'Результат', 'Status', 'status'
    ])
    
    # ВРЕМЕННЫЙ БЛОК ДИАГНОСТИКИ (удалите после проверки)
with st.expander("🔧 Диагностика (для разработчика)"):
    st.write("### Проверка колонок в загруженном файле:")
    st.write("Все колонки:", list(cases_df.columns))
    
    if 'Текущее состояние' in cases_df.columns:
        st.success("✅ Колонка 'Текущее состояние' найдена!")
        st.write("Уникальные значения в ней:", cases_df['Текущее состояние'].unique())
    else:
        st.error("❌ Колонка 'Текущее состояние' НЕ найдена!")
        st.write("Пожалуйста, проверьте названия колонок в вашем Excel-файле.")
        
    parties_col = find_column(df, [
        'Стороны', 'стороны', 'Участники', 'Parties', 'parties'
    ])
    
    # Сохраняем названия колонок в session_state
    st.session_state.description_col = description_col
    st.session_state.outcome_col = outcome_col
    st.session_state.parties_col = parties_col
    
    # Выводим информацию о найденных колонках
    st.info(f"""
    📊 **Найдено в файле:**
    - 📝 Колонка с описанием: **{description_col}** ✅
    - ⚖️ Колонка с исходами: **{outcome_col}** {'✅' if outcome_col else '❌ (не найдена)'}
    - 👥 Колонка со сторонами: **{parties_col}** {'✅' if parties_col else '⚠️ (не обязательна)'}
    """)
    
    if description_col is None:
        st.error("❌ Не найдена колонка с описаниями дел! Доступные колонки: " + ", ".join(df.columns))
        return None
    
    # Очищаем колонку с описаниями
    df = clean_description_column(df, description_col)
    
    # Если нет колонки с исходами, создаем временную
    if outcome_col is None:
        st.warning("⚠️ Колонка с исходами не найдена. Прогнозы будут без статистики.")
        df['_временный_исход'] = "неизвестно"
        st.session_state.outcome_col = '_временный_исход'
    
    st.success(f"✅ Загружено {len(df)} дел")
    
    # Показываем распределение исходов (если есть)
    if outcome_col and outcome_col in df.columns:
        outcomes = df[outcome_col].value_counts()
        st.caption(f"📊 Распределение исходов: " + ", ".join([f"{k}: {v}" for k, v in outcomes.items()]))
    
    return df


@st.cache_resource
def load_model():
    """Загружает AI-модель для преобразования текста в векторы"""
    with st.spinner("📂 Загружаю AI-модель (первые 30 секунд могут быть долгими)..."):
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data
def compute_embeddings(_model, descriptions):
    """Вычисляет векторные представления для всех описаний"""
    with st.spinner("🧠 Анализирую судебную практику (может занять 1-2 минуты)..."):
        descriptions_clean = descriptions.fillna('').astype(str).tolist()
        descriptions_clean = [desc if desc.strip() else "Описание отсутствует" for desc in descriptions_clean]
        return _model.encode(descriptions_clean, show_progress_bar=True)


# ============================================================================
# 4. ОСНОВНОЙ КОД ПРИЛОЖЕНИЯ
# ============================================================================

# ========== ШАПКА С ЛОГОТИПОМ И ЗАГОЛОВКОМ ==========
# ========== ЛОГОТИП СЛЕВА, ЗАГОЛОВОК СПРАВА ==========
# ========== ЛОГОТИП И ЗАГОЛОВОК В ОДНОЙ СТРОКЕ ==========
col_logo, col_title = st.columns([1, 10])
with col_logo:
    if os.path.exists('assets/logo.jpg'):
        st.image('assets/logo.jpg', width=120)
with col_title:
    st.markdown(
        "<h1 style='margin-bottom: 0px; margin-top: 10px;'>ЮрИнновация: Прогноз по судебным делам</h1>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='margin-top: -10px;'>Интеллектуальное право | Защита репутации | Авторское право</p>",
        unsafe_allow_html=True
    )
st.divider()

# Загружаем данные
cases_df = load_cases()
if cases_df is None:
    st.stop()

# Получаем названия колонок из session_state (безопасно)
DESC_COL = st.session_state.get('description_col', None)
OUTCOME_COL = st.session_state.get('outcome_col', None)
PARTIES_COL = st.session_state.get('parties_col', None)

# Если по какой-то причине колонка не определена, используем значение по умолчанию
if DESC_COL is None:
    DESC_COL = 'Описание'  # или то, как называется ваша колонка
    st.warning(f"⚠️ Колонка с описаниями не найдена, использую: {DESC_COL}")

# Загружаем модель
model = load_model()

# Вычисляем векторы для всех дел (один раз)
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = compute_embeddings(model, cases_df[DESC_COL])

# ============================================================================
# 5. БОКОВАЯ ПАНЕЛЬ
# ============================================================================

with st.sidebar:
    # Логотип в боковой панели
    if os.path.exists('assets/logo.jpg'):
        st.image('assets/logo.jpg', width=200)
    st.markdown("### 📊 Статистика базы")
    # ... остальной код боковой панели
    st.metric("📋 Всего дел", len(cases_df))
    
    if OUTCOME_COL and OUTCOME_COL in cases_df.columns:
        outcomes_counts = cases_df[OUTCOME_COL].value_counts()
        
        # Пытаемся найти удовлетворенные и отклоненные дела
        satisfied = 0
        rejected = 0
        for outcome, count in outcomes_counts.items():
            outcome_lower = str(outcome).lower()
            if "удовлетвор" in outcome_lower:
                satisfied += count
            elif "отказа" in outcome_lower:
                rejected += count
        
        if satisfied > 0:
            st.metric("✅ Удовлетворено", satisfied)
        if rejected > 0:
            st.metric("❌ Отказано", rejected)
    
    st.divider()
    
    st.markdown("### ⚙️ Как это работает")
    st.caption("1️⃣ Ваш текст превращается в вектор")
    st.caption("2️⃣ Ищем 8 самых похожих дел из базы")
    st.caption("3️⃣ Анализируем их исходы")
    st.caption("4️⃣ Выдаём прогноз на основе статистики")
    
    st.divider()
    
    st.markdown("### 📞 Контакты")
    st.caption("По вопросам сотрудничества:")
    st.caption("📧 https://nethouse.id/urinnovation")

# ============================================================================
# 6. ИНТЕРФЕЙС ДЛЯ ВВОДА
# ============================================================================

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📝 Опишите вашу ситуацию")
    
    input_type = st.radio(
        "Как опишете?",
        ("📖 Обычный (простым словами)", "⚖️ Юридический (с терминами)"),
        horizontal=True
    )
    
    if "Обычный" in input_type:
        placeholder = """Пример: У меня куратор скопировал фотографии с моего сайта и выложил в соцсети без моего разрешения. Могу ли я потребовать удалить их и получить компенсацию?"""
    else:
        placeholder = """Пример: Истец (правообладатель) требует признать факт незаконного использования товарного знака и взыскать компенсацию на основании ст. 1515 ГК РФ. Ответчик — владелец интернет-магазина."""
    
    user_query = st.text_area(
        "Введите описание дела:",
        height=200,
        placeholder=placeholder,
        help="Чем подробнее опишете, тем точнее будет прогноз"
    )
    
    predict_btn = st.button("🔮 Получить прогноз", type="primary", use_container_width=True)

with col_right:
    st.subheader("📊 Результат прогноза")
    result_area = st.container()

# ============================================================================
# 7. ПРОГНОЗ
# ============================================================================

if predict_btn:
    if not user_query.strip():
        st.warning("⚠️ Пожалуйста, опишите вашу ситуацию перед прогнозом.")
    else:
        with st.spinner("🧠 Анализирую судебную практику..."):
            # Превращаем запрос в вектор
            query_vec = model.encode([user_query])
            
            # Ищем похожие дела
            similarities = cosine_similarity(query_vec, st.session_state.embeddings)[0]
            top_indices = np.argsort(similarities)[-8:][::-1]
            
            similar_cases = cases_df.iloc[top_indices].copy()
            similar_cases['сходство'] = similarities[top_indices]
            
            # Анализируем исходы (используем найденную колонку)
            if OUTCOME_COL in similar_cases.columns:
                outcomes = similar_cases[OUTCOME_COL].value_counts()
                
                if not outcomes.empty:
                    main_outcome = outcomes.idxmax()
                    confidence = outcomes.max() / len(similar_cases) * 100
                else:
                    main_outcome = "неизвестно"
                    confidence = 0
            else:
                main_outcome = "неизвестно"
                confidence = 0
            
            # Определяем цвет и иконку
            outcome_color, outcome_icon = get_outcome_color(main_outcome)
            
            # Выводим результат
            with result_area:
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #2D234A 0%, #1E1A2F 100%);'>
                    <h1 style='color: {outcome_color}; margin: 0;'>{outcome_icon} {main_outcome.upper()}</h1>
                    <h3>Уверенность: {confidence:.1f}%</h3>
                    <p>на основе {len(similar_cases)} похожих дел</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ================================================================
            # ПОКАЗЫВАЕМ ПОХОЖИЕ ДЕЛА С РЕАЛЬНЫМИ НОМЕРАМИ
            # ================================================================
            with st.expander("📋 Похожие дела из практики (8 самых релевантных)"):
                for idx, (_, row) in enumerate(similar_cases.iterrows(), 1):
                    # Получаем описание
                    desc = row.get(DESC_COL, 'Описание отсутствует')
                    short_desc = desc[:200] + "..." if len(desc) > 200 else desc
                    
                    # Получаем исход
                    outcome_val = row.get(OUTCOME_COL, 'неизвестно')
                    _, outcome_emoji = get_outcome_color(outcome_val)
                    
                    # Получаем номер дела (используем универсальную функцию)
                    case_number = get_case_number(row, idx)
                    
                    # Получаем стороны (если есть)
                    parties_text = ""
                    if PARTIES_COL and PARTIES_COL in row and pd.notna(row[PARTIES_COL]):
                        parties_text = f"\n- **Стороны:** {str(row[PARTIES_COL])[:100]}"
                    
                    st.markdown(f"""
**📌 {case_number}** (сходство: {row['сходство']:.1%}) {outcome_emoji}
- **Решение:** {outcome_val}{parties_text}
- **Описание:** {short_desc}
---
""")
            
            # Совет в зависимости от уверенности
            if confidence > 70:
                st.success(f"💡 **Совет:** В {confidence:.1f}% похожих случаев суд вынес решение в пользу **{main_outcome}**. Это сильный сигнал для вашей позиции.")
            elif confidence > 40:
                st.info(f"💡 **Совет:** В {confidence:.1f}% похожих случаев суд вынес решение в пользу **{main_outcome}**. Рекомендуем изучить приведённые дела.")
            else:
                st.warning(f"💡 **Совет:** Уверенность прогноза невысока ({confidence:.1f}%). Рекомендуем проконсультироваться с юристом и добавить больше деталей в описание.")

# ============================================================================
# 8. ИНФОРМАЦИОННЫЙ РАЗДЕЛ
# ============================================================================

with st.expander("ℹ️ Как работает этот помощник"):
    st.markdown("""
**Юридический AI-помощник** анализирует ваше дело на основе реальной судебной практики.

**Как это работает:**
1. Ваш текст превращается в математический вектор (числовое представление смысла)
2. Система ищет самые похожие дела из нашей базы (8 самых релевантных)
3. Анализирует исходы этих дел (удовлетворено/отказано)
4. Выдает прогноз, основанный на статистике

**База данных:** Судебные дела по интеллектуальному праву

⚠️ **Дисклеймер:** Это демонстрационная AI-система. Прогноз не является юридической консультацией. Для принятия решений всегда консультируйтесь с квалифицированным юристом.
    """)

# ============================================================================
# 9. ОТЛАДОЧНЫЙ РЕЖИМ (раскомментируйте при необходимости)
# ============================================================================

# with st.expander("🔧 Отладка (только для разработчика)"):
#     st.write("**Названия всех колонок в файле:**")
#     st.write(list(cases_df.columns))
#     st.write(f"**Используемая колонка для описаний:** {DESC_COL}")
#     st.write(f"**Используемая колонка для исходов:** {OUTCOME_COL}")
#     st.write(f"**Используемая колонка для сторон:** {PARTIES_COL}")
