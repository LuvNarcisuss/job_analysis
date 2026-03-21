# visualization.py
import os
import warnings
from collections import Counter
import jieba
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from wordcloud import WordCloud

# 忽略所有警告信息，避免程序运行时显示警告提示
warnings.filterwarnings('ignore')

HAS_WORDCLOUD = True # 表示系统支持词云图功能
HAS_PLOTLY = True # 表示系统支持plotly功能
HAS_JIEBA = True # 表示系统支持jieba分词功能

class DataVisualization:
    """数据可视化类"""

    def __init__(self, df):
        self.df = df
        self.setup_plot_style()

        # 创建plot目录用于保存图表
        self.plot_dir = 'plot'
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def setup_plot_style(self):
        """设置绘图样式"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_palette("husl")

    def create_comprehensive_visualization(self):
        """创建综合可视化图表"""
        print("\n正在生成综合可视化图表...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('职位招聘市场综合分析', fontsize=16, fontweight='bold')

        # 1. 薪资分布
        ax1 = axes[0, 0]
        salary_data = self.df[self.df['平均薪资'].notna()]['平均薪资']
        ax1.hist(salary_data, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
        ax1.axvline(salary_data.mean(), color='red', linestyle='--', label=f'平均薪资: {salary_data.mean():.0f}元')
        ax1.axvline(salary_data.median(), color='green', linestyle='--',
                    label=f'中位薪资: {salary_data.median():.0f}元')
        ax1.set_xlabel('月薪（元）', fontsize=12)
        ax1.set_ylabel('职位数量', fontsize=12)
        ax1.set_title('薪资分布直方图', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 城市分布
        ax2 = axes[0, 1]
        city_counts = self.df['城市'].value_counts().head(10)
        colors = plt.cm.Set3(np.linspace(0, 1, len(city_counts)))
        ax2.bar(city_counts.index, city_counts.values, color=colors)
        ax2.set_xlabel('城市', fontsize=12)
        ax2.set_ylabel('职位数量', fontsize=12)
        ax2.set_title('热门城市职位分布TOP10', fontsize=14)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # 3. 工作经验要求
        ax3 = axes[0, 2]
        exp_counts = self.df['工作经验'].value_counts()
        explode = [0.1 if i == 0 else 0 for i in range(len(exp_counts))]
        ax3.pie(exp_counts.values, labels=exp_counts.index, autopct='%1.1f%%',
                colors=plt.cm.Pastel1(range(len(exp_counts))), explode=explode)
        ax3.set_title('工作经验要求分布', fontsize=14)

        # 4. 学历要求
        ax4 = axes[1, 0]
        edu_counts = self.df['学历要求'].value_counts()
        ax4.bar(edu_counts.index, edu_counts.values, color='#2ecc71', alpha=0.7)
        ax4.set_xlabel('学历要求', fontsize=12)
        ax4.set_ylabel('职位数量', fontsize=12)
        ax4.set_title('学历要求分布', fontsize=14)

        # 5. 各城市平均薪资
        ax5 = axes[1, 1]
        city_salary = self.df.groupby('城市')['平均薪资'].mean().sort_values(ascending=False).head(10)
        colors = ['#e74c3c' if x == city_salary.max() else '#f39c12' for x in city_salary.values]
        ax5.bar(city_salary.index, city_salary.values, color=colors, alpha=0.7)
        ax5.set_xlabel('城市', fontsize=12)
        ax5.set_ylabel('平均月薪（元）', fontsize=12)
        ax5.set_title('各城市平均薪资排名TOP10', fontsize=14)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

        # 6. 主要招聘公司
        ax6 = axes[1, 2]
        company_counts = self.df['公司名称'].value_counts().head(8)
        ax6.bar(company_counts.index, company_counts.values, color='#9b59b6', alpha=0.7)
        ax6.set_xlabel('公司名称', fontsize=12)
        ax6.set_ylabel('招聘职位数', fontsize=12)
        ax6.set_title('主要招聘公司TOP8', fontsize=14)
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, '综合可视化分析.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print("综合可视化图表生成完成")

    def create_position_type_analysis(self):
        """职位类型深度分析可视化"""
        print("正在生成职位类型分析图表...")

        # 职位类型统计
        position_counts = self.df['职位类型'].value_counts()
        position_salaries = self.df.groupby('职位类型')['平均薪资'].mean().sort_values(ascending=False)

        # 创建职位类型分析图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 职位数量分布
        colors1 = plt.cm.viridis(np.linspace(0, 1, len(position_counts)))
        ax1.bar(position_counts.index, position_counts.values, color=colors1)
        ax1.set_xlabel('职位类型')
        ax1.set_ylabel('职位数量')
        ax1.set_title('各职位类型数量分布')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 职位薪资分布
        colors2 = plt.cm.plasma(np.linspace(0, 1, len(position_salaries)))
        ax2.bar(position_salaries.index, position_salaries.values, color=colors2)
        ax2.set_xlabel('职位类型')
        ax2.set_ylabel('平均月薪（元）')
        ax2.set_title('各职位类型平均薪资')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, '职位类型分析.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print("职位类型分析图表生成完成")

    def create_feature_importance_plot(self, feature_importance):
        """创建特征重要性图"""
        print("正在生成特征重要性图...")

        plt.figure(figsize=(8, 6))
        plt.barh(feature_importance['特征'], feature_importance['重要性'], color='#3498db')
        plt.xlabel('特征重要性')
        plt.title('薪资预测模型特征重要性')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, '特征重要性分析.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print("特征重要性图生成完成")

    def create_salary_heatmap(self):
        """创建薪资热力图"""
        print("生成薪资热力图...")
        try:
            # 准备数据
            heatmap_data = self.df.groupby(['城市', '工作经验'])['平均薪资'].mean().unstack()

            # 过滤掉数据过少的组合
            heatmap_data = heatmap_data.dropna(how='all', axis=0).dropna(how='all', axis=1)

            if heatmap_data.empty:
                print("热力图数据不足，跳过生成")
                return

            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd',
                        cbar_kws={'label': '平均月薪（元）'})
            plt.title('各城市与工作经验对应的平均薪资热力图', fontsize=14, fontweight='bold')
            plt.xlabel('工作经验要求')
            plt.ylabel('城市')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, '薪资热力图.png'), dpi=300, bbox_inches='tight')
            plt.show()
            print("薪资热力图生成完成")
        except Exception as e:
            print(f"生成薪资热力图时出错: {e}")

    def create_salary_trend(self):
        """创建薪资趋势图"""
        print("生成薪资趋势图...")
        try:
            # 按城市和经验分组计算薪资
            trend_data = self.df.groupby(['城市', '工作经验'])['平均薪资'].agg(['mean', 'count']).reset_index()
            trend_data = trend_data[trend_data['count'] >= 3]  # 只保留有足够数据的组

            if trend_data.empty:
                print("趋势图数据不足，跳过生成")
                return

            plt.figure(figsize=(14, 8))

            # 选择主要城市
            major_cities = self.df['城市'].value_counts().head(6).index
            colors = plt.cm.tab10(range(len(major_cities)))

            for i, city in enumerate(major_cities):
                city_data = trend_data[trend_data['城市'] == city]
                if len(city_data) > 0:
                    # 确保经验顺序合理
                    exp_order = ['经验不限', '1-3年', '3-5年', '5-10年']
                    city_data['经验顺序'] = city_data['工作经验'].apply(
                        lambda x: exp_order.index(x) if x in exp_order else len(exp_order)
                    )
                    city_data = city_data.sort_values('经验顺序')

                    plt.plot(city_data['工作经验'], city_data['mean'],
                             marker='o', linewidth=2, markersize=8,
                             label=city, color=colors[i])

            plt.xlabel('工作经验要求', fontsize=12)
            plt.ylabel('平均月薪（元）', fontsize=12)
            plt.title('主要城市不同经验级别的薪资趋势', fontsize=14, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, '薪资趋势图.png'), dpi=300, bbox_inches='tight')
            plt.show()
            print("薪资趋势图生成完成")
        except Exception as e:
            print(f"生成薪资趋势图时出错: {e}")

    def create_word_cloud(self):
        """创建职位关键词词云"""
        if not HAS_WORDCLOUD:
            print("wordcloud 库未安装，跳过词云生成")
            return

        print("生成词云图...")
        try:
            # 提取职位名称中的关键词
            all_titles = ' '.join(self.df['职位名称'].dropna().astype(str))

            if HAS_JIEBA:
                # 使用jieba分词
                words = jieba.cut(all_titles)
                word_freq = Counter(words)
            else:
                # 简单的空格分词
                words = all_titles.split()
                word_freq = Counter(words)

            # 过滤停用词和短词
            stop_words = {'工程师', '开发', '高级', '资深', '岗位', '职位', '招聘', 'Python', 'Java'}
            filtered_words = {word: freq for word, freq in word_freq.items()
                              if len(word) > 1 and word not in stop_words and freq > 2}

            if not filtered_words:
                print("没有足够的关键词生成词云")
                return

            # 生成词云
            plt.figure(figsize=(12, 8))
            wordcloud = WordCloud(
                font_path='simhei.ttf' if HAS_JIEBA else None,
                width=800,
                height=600,
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate_from_frequencies(filtered_words)

            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('职位关键词词云图', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, '职位关键词词云.png'), dpi=300, bbox_inches='tight')
            plt.show()
            print("词云图生成完成")
        except Exception as e:
            print(f"生成词云图时出错: {e}")

    def create_interactive_map(self):
        """创建交互式分析图"""
        if not HAS_PLOTLY:
            print("plotly 库未安装，跳过交互式图表生成")
            return

        print("生成交互式图表...")
        try:
            # 简化版：使用条形图展示地理分布
            city_stats = self.df.groupby('城市').agg({
                '平均薪资': 'mean',
                '职位名称': 'count'
            }).rename(columns={'职位名称': '职位数量'}).reset_index()

            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('各城市职位数量', '各城市平均薪资'),
                vertical_spacing=0.1
            )

            # 职位数量条形图
            fig.add_trace(
                go.Bar(
                    x=city_stats['城市'],
                    y=city_stats['职位数量'],
                    name='职位数量',
                    marker_color='lightblue'
                ),
                row=1, col=1
            )

            # 平均薪资条形图
            fig.add_trace(
                go.Bar(
                    x=city_stats['城市'],
                    y=city_stats['平均薪资'],
                    name='平均薪资',
                    marker_color='lightcoral'
                ),
                row=2, col=1
            )

            fig.update_layout(
                height=800,
                title_text="各城市职位市场分析",
                showlegend=False
            )

            html_path = os.path.join(self.plot_dir, '职业市场交互图.html')
            fig.write_html(html_path)
            print(f"交互式图表已保存为 '{html_path}'")
        except Exception as e:
            print(f"生成交互式图表时出错: {e}")

    def create_company_analysis(self):
        """公司分析"""
        print("生成公司分析图...")
        try:
            # 分析主要招聘公司
            company_stats = self.df['公司名称'].value_counts().head(10)

            if company_stats.empty:
                print("公司分析数据不足，跳过生成")
                return

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # 公司招聘数量
            ax1.barh(company_stats.index, company_stats.values, color='steelblue')
            ax1.set_xlabel('招聘职位数量')
            ax1.set_title('TOP10招聘公司职位数量')

            # 公司平均薪资
            company_salary = self.df.groupby('公司名称')['平均薪资'].mean()
            company_salary = company_salary[company_salary.index.isin(company_stats.index)].sort_values(ascending=False)

            ax2.barh(company_salary.index, company_salary.values, color='coral')
            ax2.set_xlabel('平均月薪（元）')
            ax2.set_title('TOP10公司平均薪资')

            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, '公司分析.png'), dpi=300, bbox_inches='tight')
            plt.show()
            print("公司分析图生成完成")
        except Exception as e:
            print(f"生成公司分析图时出错: {e}")

    def create_all_basic_visualizations(self):
        """创建所有基础可视化图表"""
        print("开始生成所有基础可视化图表...")
        self.create_comprehensive_visualization()
        self.create_position_type_analysis()
        print("基础可视化图表生成完成")

    def create_all_advanced_visualizations(self):
        """创建所有高级可视化图表"""
        print("开始生成所有高级可视化图表...")
        self.create_salary_heatmap()
        self.create_salary_trend()
        self.create_company_analysis()
        self.create_word_cloud()

        if HAS_PLOTLY:
            self.create_interactive_map()

        print("高级可视化图表生成完成")


def main():
    """可视化主函数"""
    print("职位市场数据可视化项目")
    print("=" * 50)

    # 从analysis导入Analyzer
    from analysis import Analyzer

    # 初始化分析器并加载数据
    analyzer = Analyzer('zhaopin_jobs.csv')
    analyzer.load_data()
    analyzer.data_preprocessing()

    # 获取处理后的数据
    df = analyzer.df

    # 初始化可视化器
    viz = DataVisualization(df)

    # 生成所有可视化图表
    viz.create_all_basic_visualizations()
    viz.create_all_advanced_visualizations()

    # 如果有模型结果，也生成相关图表
    model_results = analyzer.build_prediction_model()
    if model_results:
        viz.create_feature_importance_plot(model_results['feature_importance'])

    print("\n所有可视化图表生成完成！")


if __name__ == "__main__":
    main()