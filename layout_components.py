# layout_components.py
# ไฟล์นี้รวบรวมฟังก์ชันสำหรับสร้างส่วนประกอบ UI ที่ใช้ซ้ำได้ในแอป

import dash_bootstrap_components as dbc
from dash import html, dash_table
import pandas as pd

# ==================================================================
# ส่วนที่ 1: ส่วนประกอบสำหรับหน้า DEEP DIVE
# ==================================================================

def create_chart_card(charts_data, recos_data, key, title):
    """สร้าง Card สำหรับแสดงกราฟ Technical Analysis"""
    chart_src = charts_data.get(key)
    reco_text = recos_data.get(key, 'N/A')
    
    card_body = html.Img(src=chart_src, width='100%') if chart_src else dbc.Alert("Chart not available.", color="warning")
    
    return dbc.Col(
        dbc.Card([
            dbc.CardHeader(f"{title} ({reco_text})"),
            dbc.CardBody(card_body)
        ]),
        width=12, lg=4, className="mb-4"
    )

def format_df_for_display(df, formats, index_name='Item'):
    """จัดรูปแบบ DataFrame สำหรับแสดงผลใน Dash Table"""
    df_display = df.copy()
    new_cols = []
    for c in df_display.columns:
        try:
            # พยายามแปลงคอลัมน์เป็นปี (YYYY) ถ้าทำได้
            new_cols.append(pd.to_datetime(c).strftime('%Y'))
        except (ValueError, TypeError, AttributeError):
            # ถ้าแปลงไม่ได้ ก็ใช้ชื่อคอลัมน์เดิม
            new_cols.append(c)
    df_display.columns = new_cols
    
    for col, fmt in formats.items():
        if col in df_display.columns: 
            df_display[col] = df_display[col].apply(lambda x: fmt.format(x) if pd.notna(x) else '-')
            
    return df_display.reset_index().rename(columns={'index': index_name})

def get_growth_style_conditions(df_display):
    """สร้างเงื่อนไขการใส่สีพื้นหลังสำหรับตาราง Growth Analysis"""
    conditions = []
    # ใช้คอลัมน์ที่ถูกแปลงเป็นปีแล้วจาก df_display
    year_cols = [c for c in df_display.columns if c.isdigit()] + ['3Y CAGR', '3Y Δ']
    for col in year_cols:
        conditions.extend([
            {'if': {'filter_query': f'{{{col}}} is num && {{{col}}} >= 0.2', 'column_id': col}, 'backgroundColor': '#C6EFCE', 'color': 'black'},
            {'if': {'filter_query': f'{{{col}}} is num && {{{col}}} < 0', 'column_id': col}, 'backgroundColor': '#F8CBAD', 'color': 'black'},
        ])
    return conditions

# <<< [CHANGED] เพิ่ม parameter `show_title=True` เข้าไป
def create_dashtable(df, title, formats, style_generator_func=None, index_name='Item', show_title=True):
    """สร้าง Dash DataTable พร้อมชื่อเรื่องและการจัดรูปแบบ"""
    if df.empty or df.shape[1] < 2:
        return html.Div([
            html.H5(title, className="mt-4") if show_title else None, # <<< ควบคุมการแสดงผล
            dbc.Alert(f"Could not load '{title}' data.", color="warning", className="mt-2")
        ], className="mb-4")
    
    df_display = format_df_for_display(df, formats, index_name)
    style_conditions = style_generator_func(df_display) if style_generator_func else []
    
    return html.Div([
        html.H5(title, className="mt-4") if show_title else None, # <<< ควบคุมการแสดงผล
        dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in df_display.columns], 
            data=df_display.to_dict('records'),
            style_cell={'textAlign': 'right'}, 
            style_header={'fontWeight': 'bold'}, 
            style_data_conditional=([{'if': {'column_id': index_name}, 'textAlign': 'left'}] or []) + style_conditions
        )
    ], className="mb-4")

# ==================================================================
# ส่วนที่ 2: ส่วนประกอบสำหรับหน้า MARKET OVERVIEW
# ==================================================================

def create_movers_table(df, title):
    """สร้างตารางสำหรับแสดง Top Movers"""
    if df is None or df.empty: 
        return dbc.Alert(f"No {title.lower()} data available.", color="light", className="mt-4")
        
    df_display = df.copy()
    df_display['Price'] = df_display['Price'].apply(lambda x: f"{x:,.2f}")
    df_display['Change %'] = df_display['Change %'].apply(lambda x: f"{x:+.2%}")
    df_display['Volume'] = df_display['Volume'].apply(lambda x: f"{x:,.0f}")
    
    return dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df_display.columns], 
        data=df_display.to_dict('records'),
        style_cell={'textAlign': 'left'}, 
        style_header={'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'filter_query': '{Change %} contains "+"', 'column_id': 'Change %'}, 'color': 'green'},
            {'if': {'filter_query': '{Change %} contains "-"', 'column_id': 'Change %'}, 'color': 'red'}
        ]
    )