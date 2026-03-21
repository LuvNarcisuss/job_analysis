# analysis.py
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 忽略所有警告信息，避免程序运行时显示警告提示
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Analyzer:
    """职位市场分析器"""

    def __init__(self, file_path):
        """初始化分析器"""
        self.file_path = file_path
        self.df = None
        self.model = None
        self.le_city = LabelEncoder()
        self.le_exp = LabelEncoder()
        self.le_edu = LabelEncoder()

    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        self.df = pd.read_csv(self.file_path, encoding='utf-8')
        print(f"成功加载数据，共{len(self.df)}条记录")

        return self.df.head()

    def data_preprocessing(self):
        """数据预处理"""
        print("\n正在进行数据预处理...")

        # 数据基本信息
        print(f"数据形状：{self.df.shape}")
        print(f"数据列：{list(self.df.columns)}")
        print("\n缺失值统计：")
        print(self.df.isnull().sum())

        # 处理薪资字段
        self.df['平均薪资'] = self.df['薪资'].apply(self.process_salary)

        # 提取城市信息
        self.df['城市'] = self.df['工作地点'].str.split('·').str[0]

        # 处理工作经验
        self.df['工作经验'] = self.df['工作经验'].apply(self.process_experience)

        # 提取职位类型
        self.df['职位类型'] = self.df['职位名称'].apply(self.extract_position_type)

        print(f"\n预处理完成，有效薪资数据：{self.df['平均薪资'].notna().sum()}条")

        return self.df

    def process_salary(self, salary):
        """处理薪资字段"""
        if pd.isna(salary) or salary == '面议':
            return np.nan

        salary_str = str(salary)

        # 处理各种薪资格式
        if '元/天' in salary_str:
            numbers = re.findall(r'(\d+\.?\d*)', salary_str)
            if numbers:
                daily_salary = float(numbers[0])
                return daily_salary * 21.75  # 转换为月薪

        # 处理范围薪资
        numbers = re.findall(r'(\d+\.?\d*)', salary_str)
        if len(numbers) >= 2:
            low = float(numbers[0])
            high = float(numbers[1])

            # 处理单位转换
            if '万' in salary_str:
                low *= 10000
                high *= 10000
            elif '千' in salary_str:
                low *= 1000
                high *= 1000

            return (low + high) / 2

        return np.nan

    def process_experience(self, exp):
        """处理工作经验字段"""
        if pd.isna(exp) or '经验不限' in str(exp) or '无经验' in str(exp):
            return '经验不限'
        return exp

    def extract_position_type(self, title):
        """提取职位类型"""
        title_lower = str(title).lower()
        if '算法' in title_lower:
            return '算法工程师'
        elif '开发' in title_lower and '测试' not in title_lower:
            return '开发工程师'
        elif '数据分析' in title_lower:
            return '数据分析师'
        elif '爬虫' in title_lower:
            return '爬虫工程师'
        elif '测试' in title_lower:
            return '测试工程师'
        elif '后端' in title_lower:
            return '后端工程师'
        elif '全栈' in title_lower:
            return '全栈工程师'
        else:
            return '其他'

    def get_analysis_data(self):
        """获取分析所需的数据"""
        return {
            'df': self.df,
            'salary_data': self.df[self.df['平均薪资'].notna()]['平均薪资'],
            'city_counts': self.df['城市'].value_counts(),
            'exp_counts': self.df['工作经验'].value_counts(),
            'edu_counts': self.df['学历要求'].value_counts(),
            'position_counts': self.df['职位类型'].value_counts(),
            'position_salaries': self.df.groupby('职位类型')['平均薪资'].mean().sort_values(ascending=False),
            'city_salary': self.df.groupby('城市')['平均薪资'].mean().sort_values(ascending=False),
            'company_counts': self.df['公司名称'].value_counts()
        }

    def generate_statistical_report(self):
        """生成统计分析报告"""
        print("\n" + "=" * 50)
        print("关键数据统计报告")
        print("=" * 50)

        analysis_data = self.get_analysis_data()
        salary_data = analysis_data['salary_data']

        print(f"总职位数量：{len(self.df):,}")
        print(f"提供薪资信息的职位：{salary_data.count():,}")
        print(f"整体平均月薪：{salary_data.mean():.2f}元")
        print(f"薪资中位数：{salary_data.median():.2f}元")
        print(f"最高月薪：{salary_data.max():.2f}元")
        print(f"最低月薪：{salary_data.min():.2f}元")
        print(f"薪资标准差：{salary_data.std():.2f}元")

        print("\n地域分析：")
        city_counts = analysis_data['city_counts'].head()
        for city, count in city_counts.items():
            percentage = (count / len(self.df)) * 100
            city_salary = self.df[self.df['城市'] == city]['平均薪资'].mean()
            print(f"  {city}: {count}个职位 ({percentage:.1f}%), 平均薪资: {city_salary:.0f}元")

        print("\n经验要求分析：")
        for exp, count in analysis_data['exp_counts'].items():
            percentage = (count / len(self.df)) * 100
            print(f"  {exp}: {count}个职位 ({percentage:.1f}%)")

    def build_prediction_model(self):
        """构建薪资预测模型"""
        print("\n正在构建薪资预测模型...")

        # 准备建模数据
        model_df = self.df[['平均薪资', '城市', '工作经验', '学历要求']].copy()
        model_df = model_df.dropna()

        if len(model_df) < 100:
            print("数据量不足，跳过建模环节")
            return None

        # 编码分类变量
        model_df['城市编码'] = self.le_city.fit_transform(model_df['城市'])
        model_df['经验编码'] = self.le_exp.fit_transform(model_df['工作经验'])
        model_df['学历编码'] = self.le_edu.fit_transform(model_df['学历要求'])

        # 特征和目标变量
        X = model_df[['城市编码', '经验编码', '学历编码']]
        y = model_df['平均薪资']

        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 训练模型
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # 预测和评估
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("模型训练完成！")
        print(f"平均绝对误差 (MAE): {mae:.2f}元")
        print(f"R² 分数: {r2:.4f}")

        # 特征重要性
        feature_importance = pd.DataFrame({
            '特征': ['城市', '工作经验', '学历要求'],
            '重要性': self.model.feature_importances_
        }).sort_values('重要性', ascending=False)

        print("\n特征重要性排序：")
        print(feature_importance)

        return {
            'model': self.model,
            'feature_importance': feature_importance,
            'metrics': {'mae': mae, 'r2': r2}
        }


    def predict_salary(self, city, experience, education):
        """预测薪资 - 使用训练好的H5模型文件进行预测"""

        # 添加模型缓存属性
        if not hasattr(self, '_cached_model'):
            self._cached_model = None
            self._cached_model_path = None

        # 检查是否存在预先训练好的H5模型文件
        model_path = 'salary_predictor_model.h5'

        if os.path.exists(model_path):
            try:
                # 只在首次或模型路径改变时加载模型
                if self._cached_model is None or self._cached_model_path != model_path:
                    import tensorflow as tf
                    self._cached_model = tf.keras.models.load_model(model_path)
                    self._cached_model_path = model_path
                    print(f"成功加载预训练模型: {model_path}")

                # 获取城市等级映射
                city_tier_map = self._get_city_tier_map()
                city_tier = city_tier_map.get(city, city_tier_map.get('default', 0.3))

                # 学历映射
                education_mapping = {
                    '高中': 1,
                    '大专': 2,
                    '本科': 3,
                    '硕士': 4,
                    '博士': 5,
                    '学历不限': 0
                }
                education_code = education_mapping.get(education, 0)

                # 工作经验处理
                experience_years = self._convert_experience_to_years(experience)

                # 构造特征向量
                features = np.array([[city_tier, education_code, experience_years]])

                # 使用缓存的模型进行预测
                predicted_salary = self._cached_model.predict(features, verbose=0)[0][0]

                return predicted_salary

            except Exception as e:
                print(f"使用H5模型预测失败: {e}")
                # 回退到原有模型预测
                return self._predict_with_existing_model(city, experience, education)
        else:
            print(f"未找到预训练模型文件: {model_path}")
            # 使用现有的随机森林模型进行预测
            return self._predict_with_existing_model(city, experience, education)


    def _predict_with_existing_model(self, city, experience, education):
        """使用现有模型进行预测（回退方案）"""
        if self.model is None:
            print("请先训练模型！")
            return None

        try:
            # 检查输入值是否在训练数据中存在
            if city not in self.le_city.classes_:
                print(f"警告: 城市 '{city}' 不在训练数据中，将使用默认值")
                city = self.le_city.classes_[0]

            if experience not in self.le_exp.classes_:
                print(f"警告: 工作经验 '{experience}' 不在训练数据中，将使用默认值")
                experience = self.le_exp.classes_[0]

            if education not in self.le_edu.classes_:
                print(f"警告: 学历 '{education}' 不在训练数据中，将使用默认值")
                education = self.le_edu.classes_[0]

            # 对输入进行编码
            city_encoded = self.le_city.transform([city])[0]
            exp_encoded = self.le_exp.transform([experience])[0]
            edu_encoded = self.le_edu.transform([education])[0]

            # 构造特征向量并预测
            features = np.array([[city_encoded, exp_encoded, edu_encoded]])
            predicted_salary = self.model.predict(features)[0]

            return predicted_salary
        except Exception as e:
            print(f"预测失败：{e}")
            return None

    def _get_city_tier_map(self):
        """获取城市等级映射（与salary_predictor.py中一致）"""
        city_tier_map = {}

        # 一线城市
        tier_1 = ['北京', '上海', '深圳', '广州']
        # 新一线城市
        tier_1_5 = ['成都', '杭州', '重庆', '西安', '苏州', '武汉', '南京', '天津',
                    '郑州', '长沙', '东莞', '佛山', '宁波', '青岛', '沈阳']
        # 二线城市
        tier_2 = ['合肥', '昆明', '无锡', '厦门', '济南', '福州', '温州', '金华',
                  '嘉兴', '惠州', '中山', '保定', '邯郸', '临沂', '唐山', '海口',
                  '绍兴', '珠海', '贵阳', '南昌', '南宁', '泉州', '常州', '南通']
        # 三线及以下
        tier_3 = ['兰州', '银川', '西宁', '呼和浩特', '包头', '乌鲁木齐', '拉萨',
                  '三亚', '汕头', '绵阳', '洛阳', '襄阳', '宜昌', '岳阳', '衡阳',
                  '柳州', '珠海', '中山', '江门', '湛江', '茂名', '肇庆', '清远',
                  '韶关', '阳江', '云浮', '河源', '梅州', '汕尾', '潮州', '揭阳']

        # 创建城市等级映射
        for city in tier_1:
            city_tier_map[city] = 1.0
        for city in tier_1_5:
            city_tier_map[city] = 0.8
        for city in tier_2:
            city_tier_map[city] = 0.6
        for city in tier_3:
            city_tier_map[city] = 0.4

        # 默认值
        city_tier_map['default'] = 0.3

        return city_tier_map

    def _convert_experience_to_years(self, exp):
        """将工作经验转换为数值（年）"""
        if pd.isna(exp):
            return 0

        exp_str = str(exp).lower()

        # 经验映射
        experience_map = {
            '经验不限': 0,
            '无经验': 0,
            '在校生': 0,
            '1年以下': 0.5,
            '1-3年': 2,
            '3-5年': 4,
            '5-10年': 7.5,
            '10年以上': 12
        }

        # 查找匹配
        for key, value in experience_map.items():
            if key in exp_str:
                return value

        # 尝试提取数字
        numbers = re.findall(r'\d+', exp_str)
        if numbers:
            if len(numbers) >= 2:
                return (float(numbers[0]) + float(numbers[1])) / 2
            else:
                return float(numbers[0])

        return 0



    def generate_final_report(self):
        """生成最终分析报告"""
        print("\n" + "=" * 60)
        print("Python职位招聘市场分析最终报告")
        print("=" * 60)

        analysis_data = self.get_analysis_data()
        salary_data = analysis_data['salary_data']

        print("\n一、市场概况")
        print(f"• 总职位数量：{len(self.df):,}个")
        print(f"• 有效薪资样本：{len(salary_data):,}个")
        print(f"• 市场平均月薪：{salary_data.mean():.0f}元")

        print("\n二、地域分布特征")
        top_cities = analysis_data['city_counts'].head(3)
        for city in top_cities.index:
            count = top_cities[city]
            salary = self.df[self.df['城市'] == city]['平均薪资'].mean()
            print(f"• {city}：{count}个职位，平均薪资{salary:.0f}元")

        print("\n三、人才需求特征")
        print(f"• 主要经验要求：{analysis_data['exp_counts'].index[0]}")
        print(f"• 主要学历要求：{analysis_data['edu_counts'].index[0]}")

        print("\n四、职位类型分布")
        top_positions = analysis_data['position_counts'].head(3)
        for position in top_positions.index:
            count = top_positions[position]
            print(f"• {position}：{count}个职位")

        print("\n五、业务建议")
        print("• 求职者：重点关注算法和开发方向，目标城市选择北京、上海、深圳")
        print("• 企业：在二三线城市可提供有竞争力的薪资吸引人才")
        print("• 教育机构：加强3-5年经验人才的实战能力培养")


def main():
    """分析主函数"""
    print("招聘市场分析")

    # 初始化分析器
    analyzer = Analyzer('zhaopin_jobs.csv')

    # 执行分析流程
    analyzer.load_data()
    analyzer.data_preprocessing()

    # 生成统计分析报告
    analyzer.generate_statistical_report()

    # 构建预测模型
    model_results = analyzer.build_prediction_model()

    # 生成最终报告
    analyzer.generate_final_report()

    # 使用固定的示例数据进行预测
    example_cases = [
        {'city': '北京', 'experience': 3.0, 'education': '本科'},
        {'city': '上海', 'experience': 5.0, 'education': '硕士'},
        {'city': '深圳', 'experience': 1.0, 'education': '大专'},
        {'city': '杭州', 'experience': 8.0, 'education': '本科'},
        {'city': '成都', 'experience': 1.0, 'education': '本科'},
    ]

    print("\n" + "=" * 60)
    print("示例薪资预测:")
    print("=" * 60)

    for i, case in enumerate(example_cases, 1):
        try:
            predicted_salary = analyzer.predict_salary(
                city=case['city'],
                experience=case['experience'],
                education=case['education']
            )

            if predicted_salary is not None:
                print(f"\n预测案例{i}:", end=" ")
                print(f"{case['city']} {case['education']}学历", end=" ")
                print(f"工作经验{case['experience']:.1f}年")
                print("-" * 30)
                print(f"   预测平均月薪: {predicted_salary:,.0f} 元")

                # 提供市场建议
                if predicted_salary < 8000:
                    print("   - 市场建议: 该薪资处于初级水平，建议关注技能提升")
                elif predicted_salary < 15000:
                    print("   - 市场建议: 该薪资处于中等水平，具备市场竞争力")
                elif predicted_salary < 25000:
                    print("   - 市场建议: 该薪资处于中高水平，是资深人才的合理期望")
                else:
                    print("   - 市场建议: 该薪资处于高水平，适合专家级人才")
            else:
                print(f"\n{i}. 预测案例: {case['city']} - 预测失败")

        except Exception as e:
            print(f"   预测错误: {e}")

    print("\n数据分析完成！")



if __name__ == "__main__":
    main()