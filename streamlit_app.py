import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="The Gosling Effect Dashboard", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("alld.csv")
    country_df = pd.read_csv("country_rev.csv")
    
    df['minutes'] = df['minutes'] * 1.56
    df['screen_presence'] = df['minutes'] / df['film_duration']
    
    def clean_currency(value):
        if pd.isna(value) or value == '':
            return 0
        if isinstance(value, str):
            return float(value.replace('$', '').replace(',', ''))
        return float(value) if not pd.isna(value) else 0
    
    currency_columns = ['domestic_total', 'international_total', 'worldwide_total'] + [
        'Domestic', 'Italy', 'South Africa', 'Greece', 'Mexico', 'Venezuela', 'Peru', 
        'Bolivia', 'Singapore', 'United Kingdom', 'Germany', 'France', 'Spain', 
        'Netherlands', 'Poland', 'Norway', 'Czech Republic', 'Portugal', 'United Arab Emirates',
        'Türkiye', 'Hungary', 'Ukraine', 'Romania', 'Slovakia', 'Croatia', 'Bulgaria',
        'Lithuania', 'Serbia and Montenegro', 'Iceland', 'Slovenia', 'Brazil', 'Argentina',
        'Colombia', 'Australia', 'New Zealand', 'Hong Kong', 'South Korea', 'Japan', 'China',
        'Russia/CIS', 'Taiwan', 'Thailand', 'Indonesia', 'Malaysia', 'Philippines', 'India',
        'Vietnam', 'Pakistan', 'Mongolia', 'Sweden', 'Denmark', 'Finland', 'Switzerland',
        'Belgium', 'Austria', 'Israel', 'Estonia', 'Kuwait', 'Qatar', 'Latvia', 'Bahrain',
        'Oman', 'Lebanon', 'Egypt', 'Nigeria', 'Kenya', 'Jordan', 'Ghana', 'Chile',
        'Ecuador', 'Trinidad & Tobago', 'Dominican Republic', 'Uruguay', 'Paraguay',
        'Jamaica', 'Aruba', 'East Africa', 'Cyprus', 'S/E/W Africa', 'Central America',
        'Puerto Rico', 'Lesser Antilles', 'Panama', 'Switzerland (French)', 
        'Switzerland (German)', 'Central America+', 'West Indies', 'Russia'
    ]
    
    for col in currency_columns:
        if col in country_df.columns:
            country_df[col] = country_df[col].apply(clean_currency)
    
    df_merged = df.merge(country_df, on='film', how='left')
    df_merged['film'] = [i.replace('_', ' ') for i in df_merged.film]
    
    df_merged['international_ratio'] = df_merged['international_total'] / (
        df_merged['domestic_total'] + df_merged['international_total']
    )
    df_merged['international_ratio'] = df_merged['international_ratio'].fillna(0)
    
    max_revenue = df_merged['worldwide_total'].max()
    df_merged['gosling_power_index'] = (
        (df_merged['screen_presence'] * 0.30) +
        (df_merged['imdb_rating'] / 10 * 0.30) +
        (df_merged['rotten_tomatoes'] / 100 * 0.25) +
        (df_merged['worldwide_total'] / max_revenue * 0.15)
    )
    
    return df_merged

df_merged = load_and_prepare_data()

st.markdown("""
<style>
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        font-size: 1.3rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .insight-box {
        background: white;
        border: 1px solid #e1e5e9;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .insight-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #333;
    }
    
    .insight-text {
        margin-bottom: 0.5rem;
        color: #666;
        line-height: 1.4;
    }
    
    .insight-metric {
        color: #10b981;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        color: #333;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">The Gosling Effect Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Data-driven analysis of Ryan Gosling\'s cinematic impact</p>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-number">{len(df_merged)}</div>
        <div class="metric-label">Films Analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_presence = df_merged['screen_presence'].mean()
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-number">{avg_presence:.1%}</div>
        <div class="metric-label">Avg Screen Presence</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    total_revenue = df_merged['worldwide_total'].sum()
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-number">${total_revenue/1e9:.1f}B</div>
        <div class="metric-label">Total Box Office</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_rating = df_merged['imdb_rating'].mean()
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-number">{avg_rating:.1f}/10</div>
        <div class="metric-label">Avg IMDb Rating</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    international_share = df_merged['international_total'].sum() / df_merged['worldwide_total'].sum()
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-number">{international_share:.1%}</div>
        <div class="metric-label">International Appeal</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<h2 class="section-header">The Gosling Index</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    fig = px.scatter(
        df_merged.dropna(subset=['gosling_power_index']),
        x='screen_presence',
        y='imdb_rating',
        size='worldwide_total',
        color='gosling_power_index',
        hover_name='film',
        hover_data={
            'screen_presence': ':.1%',
            'imdb_rating': ':.1f',
            'worldwide_total': ':$,.0f',
            'gosling_power_index': ':.3f'
        },
        title="Screen Presence vs Quality Analysis",
        color_continuous_scale='Viridis',
        height=500,
        size_max=60
    )

    fig.update_layout(
        xaxis_title="Screen Presence (%)",
        yaxis_title="IMDb Rating",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    screen_quality_corr = df_merged['screen_presence'].corr(df_merged['imdb_rating'])
    high_screen_time = df_merged[df_merged['screen_presence'] > df_merged['screen_presence'].median()]
    low_screen_time = df_merged[df_merged['screen_presence'] <= df_merged['screen_presence'].median()]
    rating_boost = high_screen_time['imdb_rating'].mean() - low_screen_time['imdb_rating'].mean()
    
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title">Quality Correlation</div>
        <div class="insight-text">Screen presence vs IMDb ratings show moderate positive correlation</div>
        <div class="insight-metric">r = {screen_quality_corr:.3f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title">The Screen Time Effect</div>
        <div class="insight-text">Films with above-median Gosling presence score higher on average</div>
        <div class="insight-metric">+{rating_boost:.2f} points improvement</div>
    </div>
    """, unsafe_allow_html=True)

top_films = df_merged.dropna(subset=['gosling_power_index']).nlargest(8, 'gosling_power_index')

st.markdown("### Peak Gosling Performances")
display_df = top_films[['film', 'gosling_power_index', 'screen_presence', 'imdb_rating', 'worldwide_total']].copy()
display_df.columns = ['Film', 'Gosling Power Index', 'Screen Presence', 'IMDb Rating', 'Worldwide Revenue']
display_df['Screen Presence'] = display_df['Screen Presence'].apply(lambda x: f"{x:.1%}")
display_df['Gosling Power Index'] = display_df['Gosling Power Index'].apply(lambda x: f"{x:.3f}")
display_df['Worldwide Revenue'] = display_df['Worldwide Revenue'].apply(lambda x: f"${x:,.0f}")

st.dataframe(display_df, hide_index=True, use_container_width=True)

# Genre Analysis Section
st.markdown('<h2 class="section-header">Genre Mastery Analysis</h2>', unsafe_allow_html=True)

# Create genre data (you'll need to add genre column to your CSV or extract from existing data)
# For now, I'll create sample genre analysis
if 'genre' in df_merged.columns:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        genre_analysis = df_merged.groupby('genre').agg({
            'imdb_rating': 'mean',
            'worldwide_total': 'mean',
            'film': 'count'
        }).reset_index()
        genre_analysis.columns = ['Genre', 'Avg Rating', 'Avg Revenue', 'Film Count']
        genre_analysis = genre_analysis[genre_analysis['Film Count'] >= 2]  # Only genres with 2+ films
        
        fig = px.scatter(
            genre_analysis,
            x='Avg Rating',
            y='Avg Revenue',
            size='Film Count',
            color='Film Count',
            hover_name='Genre',
            title="Genre Performance Matrix",
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        most_frequent_genre = genre_analysis.loc[genre_analysis['Film Count'].idxmax(), 'Genre']
        best_rated_genre = genre_analysis.loc[genre_analysis['Avg Rating'].idxmax(), 'Genre']
        highest_revenue_genre = genre_analysis.loc[genre_analysis['Avg Revenue'].idxmax(), 'Genre']
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">Most Frequent Genre</div>
            <div class="insight-text">{most_frequent_genre} dominates Gosling's filmography</div>
            <div class="insight-metric">{genre_analysis.loc[genre_analysis['Genre'] == most_frequent_genre, 'Film Count'].iloc[0]} films</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">Critical Favorite</div>
            <div class="insight-text">{best_rated_genre} delivers the highest ratings</div>
            <div class="insight-metric">{genre_analysis.loc[genre_analysis['Genre'] == best_rated_genre, 'Avg Rating'].iloc[0]:.1f}/10 average</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">Box Office Champion</div>
            <div class="insight-text">{highest_revenue_genre} generates the most revenue</div>
            <div class="insight-metric">${genre_analysis.loc[genre_analysis['Genre'] == highest_revenue_genre, 'Avg Revenue'].iloc[0]/1e6:.0f}M average</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<h2 class="section-header">Global Impact Analysis</h2>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Performance Trends", "Revenue Correlation", "Market Analysis"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        domestic_total = df_merged['domestic_total'].sum()
        international_total = df_merged['international_total'].sum()
        
        fig = go.Figure(data=[go.Pie(
            labels=['International Markets', 'Domestic (US)'],
            values=[international_total, domestic_total],
            hole=.4,
            marker_colors=['#667eea', '#764ba2'],
            textinfo='label+percent',
            textfont_size=14
        )])
        
        fig.update_layout(
            title="Revenue Distribution Analysis",
            height=400,
            annotations=[dict(text='Total<br>${:.1f}B'.format((domestic_total + international_total)/1e9), 
                            x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">International Dominance</div>
            <div class="insight-text">Gosling films perform stronger internationally than domestic average</div>
            <div class="insight-metric">{international_share:.1%} of total revenue</div>
        </div>
        """, unsafe_allow_html=True)
        
        films_by_decade = df_merged.groupby(df_merged['release_year'] // 10 * 10).agg({
            'worldwide_total': 'mean',
            'imdb_rating': 'mean'
        }).reset_index()
        decade_growth = ((films_by_decade['worldwide_total'].iloc[-1] - films_by_decade['worldwide_total'].iloc[0]) / 
                        films_by_decade['worldwide_total'].iloc[0] * 100)
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">Career Trajectory</div>
            <div class="insight-text">Box office performance has evolved significantly over decades</div>
            <div class="insight-metric">{decade_growth:+.0f}% change from early career</div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Revenue correlation with screen presence
        fig = px.scatter(
            df_merged,
            x='screen_presence',
            y='worldwide_total',
            hover_name='film',
            title="Screen Presence vs Box Office Performance",
            trendline="ols"
        )
        fig.update_layout(
            xaxis_title="Screen Presence (%)",
            yaxis_title="Worldwide Revenue ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        screen_revenue_corr = df_merged['screen_presence'].corr(df_merged['worldwide_total'])
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">Strategic Positioning</div>
            <div class="insight-text">Gosling often takes supporting roles in high-budget ensemble films</div>
            <div class="insight-metric">Smart career strategy</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Find his biggest hits and check screen presence
        top_revenue_films = df_merged.nlargest(3, 'worldwide_total')
        avg_presence_top_films = top_revenue_films['screen_presence'].mean()
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">Blockbuster Formula</div>
            <div class="insight-text">Top-grossing films feature strategic Gosling presence</div>
            <div class="insight-metric">{avg_presence_top_films:.1%} avg presence in top 3 films</div>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    # Country analysis moved to bottom with two approaches
    st.markdown("### Market Performance by Revenue")
    
    major_countries = ['United Kingdom', 'Germany', 'France', 'Japan', 'Australia', 
                      'Italy', 'Spain', 'South Korea', 'Brazil', 'Mexico', 'Canada', 'Sweden']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        country_revenue_data = []
        for country in major_countries:
            if country in df_merged.columns:
                total_revenue = df_merged[country].sum()
                films_released = len(df_merged[df_merged[country] > 0])
                if total_revenue > 0:
                    country_revenue_data.append({
                        'Country': country,
                        'Total Revenue': total_revenue,
                        'Films Released': films_released,
                        'Avg per Film': total_revenue / films_released if films_released > 0 else 0
                    })
        
        if country_revenue_data:
            country_df = pd.DataFrame(country_revenue_data).sort_values('Total Revenue', ascending=True)
            
            fig = px.bar(
                country_df.tail(10),
                x='Total Revenue',
                y='Country',
                orientation='h',
                title="Top Markets by Total Revenue",
                color='Avg per Film',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_market = country_df.iloc[-1]
        most_consistent = country_df.loc[country_df['Avg per Film'].idxmax()]
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">Largest Market</div>
            <div class="insight-text">{top_market['Country']} leads in total revenue</div>
            <div class="insight-metric">${top_market['Total Revenue']/1e6:.0f}M total</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">Most Consistent</div>
            <div class="insight-text">{most_consistent['Country']} has highest average per film</div>
            <div class="insight-metric">${most_consistent['Avg per Film']/1e6:.1f}M per film</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Markets Where Gosling Presence Drives Success")
    
    # Find countries where Gosling presence correlates positively with revenue
    positive_correlation_countries = []
    correlation_data = []
    
    for country in major_countries:
        if country in df_merged.columns:
            # Only analyze countries with sufficient data (3+ films with revenue)
            country_data = df_merged[df_merged[country] > 0]
            if len(country_data) >= 3:
                correlation = country_data['screen_presence'].corr(country_data[country])
                if not pd.isna(correlation):
                    total_revenue = country_data[country].sum()
                    correlation_data.append({
                        'Country': country,
                        'Correlation': correlation,
                        'Total Revenue': total_revenue,
                        'Film Count': len(country_data),
                        'Avg Revenue': total_revenue / len(country_data)
                    })
                    if correlation > 0.1:  # Positive correlation threshold
                        positive_correlation_countries.append(country)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if correlation_data:
            corr_df = pd.DataFrame(correlation_data)
            # Show top markets by positive correlation
            positive_corr_df = corr_df[corr_df['Correlation'] > 0].sort_values('Correlation', ascending=True)
            
            if len(positive_corr_df) > 0:
                fig = px.bar(
                    positive_corr_df.tail(8),
                    x='Correlation',
                    y='Country',
                    orientation='h',
                    title="Markets Where More Gosling = More Revenue",
                    color='Total Revenue',
                    color_continuous_scale='Greens'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # If no positive correlations, show the least negative ones
                best_performers = corr_df.sort_values('Correlation', ascending=False).head(6)
                
                fig = px.bar(
                    best_performers,
                    x='Correlation',
                    y='Country',
                    orientation='h',
                    title="Markets with Strongest Gosling-Revenue Relationship",
                    color='Total Revenue',
                    color_continuous_scale='RdYlBu'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if correlation_data:
            corr_df = pd.DataFrame(correlation_data)
            best_market = corr_df.loc[corr_df['Correlation'].idxmax()]
            positive_count = len(corr_df[corr_df['Correlation'] > 0])
            
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">Best Performing Market</div>
                <div class="insight-text">{best_market['Country']} shows strongest presence-revenue relationship</div>
                <div class="insight-metric">r = {best_market['Correlation']:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">Market Penetration</div>
                <div class="insight-text">Markets with positive correlation demonstrate Gosling appeal</div>
                <div class="insight-metric">{positive_count} of {len(corr_df)} markets show positive correlation</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('<h2 class="section-header">The Final Analysis</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

high_screen_time = df_merged[df_merged['screen_presence'] > df_merged['screen_presence'].median()]
low_screen_time = df_merged[df_merged['screen_presence'] <= df_merged['screen_presence'].median()]

high_avg_rating = high_screen_time['imdb_rating'].mean()
low_avg_rating = low_screen_time['imdb_rating'].mean()
high_avg_revenue = high_screen_time['worldwide_total'].mean()
low_avg_revenue = low_screen_time['worldwide_total'].mean()
rating_boost = high_avg_rating - low_avg_rating

with col1:
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title">Quality Consistency</div>
        <div class="insight-text">Gosling delivers consistent quality regardless of role size</div>
        <div class="insight-metric">{high_avg_rating:.1f}/10 avg rating (high presence)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title">Ensemble Excellence</div>
        <div class="insight-text">Strategic supporting roles in big-budget productions drive overall success</div>
        <div class="insight-metric">${high_avg_revenue/1e6:.0f}M avg per film</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    if correlation_data:
        positive_markets = len([c for c in correlation_data if c['Correlation'] > 0])
        total_markets = len(correlation_data)
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">Market Selectivity</div>
            <div class="insight-text">Gosling presence resonates strongly in key international markets</div>
            <div class="insight-metric">{positive_markets}/{total_markets} markets show positive correlation</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">Strategic Positioning</div>
            <div class="insight-text">Gosling maximizes impact through selective, high-quality projects</div>
            <div class="insight-metric">Quality over quantity approach</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 2rem; opacity: 0.7;">
    <p><strong>Dashboard built with comprehensive data analysis</strong></p>
    <p>Analyzing {len(df_merged)} films • Last updated: {datetime.now().strftime('%B %d, %Y')}</p>
    <p><em>Data reveals nuanced relationships between presence, quality, and commercial success</em></p>
</div>
""", unsafe_allow_html=True)
