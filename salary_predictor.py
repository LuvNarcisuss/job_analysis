# salary_predictor.py
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import warnings

warnings.filterwarnings('ignore')


class SalaryPredictor:
    """薪资预测模型类"""

    def __init__(self):
        self.location_encoder = LabelEncoder()
        self.education_encoder = LabelEncoder()
        self.education_mapping = None
        self.experience_scaler = StandardScaler()
        self.salary_scaler = StandardScaler()
        self.model = None
        self.city_tier_map = {}

    def clean_data(self, df):
        """数据清洗和预处理"""
        print("开始数据清洗...")

        # 1. 薪资转换函数
        def convert_salary(salary_str):
            """将薪资字符串转换为数值（月薪，单位：元）"""
            if pd.isna(salary_str) or str(salary_str).strip() in ['面议', '']:
                return None

            salary_str = str(salary_str)

            # 处理面议情况
            if '面议' in salary_str:
                return None

            # 提取数字部分
            numbers = re.findall(r'\d+\.?\d*', salary_str)
            numbers = [float(n) for n in numbers]

            if not numbers:
                return None

            # 判断单位（万/元）
            if '万' in salary_str:
                # 万元转元
                numbers = [n * 10000 for n in numbers]

            # 判断是否包含13薪、14薪等
            if '13薪' in salary_str:
                # 转换为月薪（年薪/12）
                numbers = [n / 12 * 13 for n in numbers]
            elif '14薪' in salary_str:
                numbers = [n / 12 * 14 for n in numbers]
            elif '15薪' in salary_str:
                numbers = [n / 12 * 15 for n in numbers]
            elif '16薪' in salary_str:
                numbers = [n / 12 * 16 for n in numbers]

            # 返回薪资范围（最小值和最大值）
            if len(numbers) >= 2:
                return [min(numbers), max(numbers)]
            else:
                return [numbers[0], numbers[0]]

        # 2. 处理薪资列
        df['salary_clean'] = df['薪资'].apply(convert_salary)

        # 移除薪资为空的行
        df = df.dropna(subset=['salary_clean'])

        # 提取薪资最小值和最大值
        df['salary_min'] = df['salary_clean'].apply(lambda x: x[0])
        df['salary_max'] = df['salary_clean'].apply(lambda x: x[1])
        df['salary_avg'] = (df['salary_min'] + df['salary_max']) / 2

        # 3. 处理工作地点
        def extract_location_info(location_str):
            """从工作地点字符串中提取城市和区域"""
            if pd.isna(location_str):
                return None, None

            parts = str(location_str).split('·')
            if len(parts) >= 1:
                city = parts[0].strip()
            else:
                city = location_str.strip()

            # 简化城市名称
            city_simplified = self.simplify_city_name(city)
            return city_simplified

        df['city'] = df['工作地点'].apply(extract_location_info)

        # 4. 处理工作经验
        def convert_experience(exp_str):
            """将工作经验转换为数值（年）"""
            if pd.isna(exp_str):
                return 0

            exp_str = str(exp_str).lower()

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

        df['experience_years'] = df['工作经验'].apply(convert_experience)

        # 5. 处理学历要求
        def simplify_education(edu_str):
            """简化学历要求"""
            if pd.isna(edu_str):
                return '学历不限'

            edu_str = str(edu_str).lower()

            if '博士' in edu_str:
                return '博士'
            elif '硕士' in edu_str:
                return '硕士'
            elif '本科' in edu_str:
                return '本科'
            elif '大专' in edu_str:
                return '大专'
            elif '高中' in edu_str or '中专' in edu_str:
                return '高中'
            else:
                return '学历不限'

        df['education_simple'] = df['学历要求'].apply(simplify_education)

        # 6. 定义城市等级
        self.define_city_tiers()

        # 7. 移除异常值（薪资过高或过低）
        q1 = df['salary_avg'].quantile(0.01)
        q99 = df['salary_avg'].quantile(0.99)
        df = df[(df['salary_avg'] >= q1) & (df['salary_avg'] <= q99)]

        print(f"数据清洗完成，剩余 {len(df)} 条记录")
        return df

    def simplify_city_name(self, city):
        """简化城市名称"""
        # 常见城市名称映射
        city_mapping = {
            '北京': '北京',
            '上海': '上海',
            '深圳': '深圳',
            '广州': '广州',
            '杭州': '杭州',
            '成都': '成都',
            '武汉': '武汉',
            '南京': '南京',
            '苏州': '苏州',
            '西安': '西安',
            '重庆': '重庆',
            '天津': '天津',
            '郑州': '郑州',
            '长沙': '长沙',
            '合肥': '合肥',
            '济南': '济南',
            '青岛': '青岛',
            '大连': '大连',
            '沈阳': '沈阳',
            '长春': '长春',
            '哈尔滨': '哈尔滨',
            '福州': '福州',
            '厦门': '厦门',
            '珠海': '珠海',
            '东莞': '东莞',
            '佛山': '佛山',
            '中山': '中山',
            '惠州': '惠州',
            '宁波': '宁波',
            '温州': '温州',
            '金华': '金华',
            '嘉兴': '嘉兴',
            '绍兴': '绍兴',
            '台州': '台州',
            '湖州': '湖州',
            '常州': '常州',
            '无锡': '无锡',
            '徐州': '徐州',
            '南通': '南通',
            '扬州': '扬州',
            '盐城': '盐城',
            '泰州': '泰州',
            '镇江': '镇江',
            '石家庄': '石家庄',
            '太原': '太原',
            '呼和浩特': '呼和浩特',
            '包头': '包头',
            '鄂尔多斯': '鄂尔多斯',
            '乌海': '乌海',
            '赤峰': '赤峰',
            '通辽': '通辽',
            '兰州': '兰州',
            '西宁': '西宁',
            '银川': '银川',
            '乌鲁木齐': '乌鲁木齐',
            '克拉玛依': '克拉玛依',
            '吐鲁番': '吐鲁番',
            '哈密': '哈密',
            '昌吉': '昌吉',
            '博尔塔拉': '博尔塔拉',
            '巴音郭楞': '巴音郭楞',
            '阿克苏': '阿克苏',
            '克孜勒苏': '克孜勒苏',
            '喀什': '喀什',
            '和田': '和田',
            '伊犁': '伊犁',
            '塔城': '塔城',
            '阿勒泰': '阿勒泰',
            '石河子': '石河子',
            '阿拉尔': '阿拉尔',
            '图木舒克': '图木舒克',
            '五家渠': '五家渠',
            '北屯': '北屯',
            '铁门关': '铁门关',
            '双河': '双河',
            '可克达拉': '可克达拉',
            '昆玉': '昆玉',
            '胡杨河': '胡杨河',
            '新星': '新星',
            '白杨': '白杨',
            '香港': '香港',
            '澳门': '澳门',
            '台湾': '台湾'
        }

        # 查找匹配
        for key, value in city_mapping.items():
            if key in city:
                return value

        # 如果未匹配到，返回原城市名称
        return city

    def define_city_tiers(self):
        """定义城市等级"""
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
            self.city_tier_map[city] = 1.0
        for city in tier_1_5:
            self.city_tier_map[city] = 0.8
        for city in tier_2:
            self.city_tier_map[city] = 0.6
        for city in tier_3:
            self.city_tier_map[city] = 0.4

        # 默认值
        self.city_tier_map['default'] = 0.3

    def prepare_features(self, df):
        """准备特征数据"""
        print("准备特征数据...")

        # 1. 处理城市特征（使用城市等级）
        df['city_tier'] = df['city'].apply(
            lambda x: self.city_tier_map.get(x, self.city_tier_map['default'])
        )

        # 2. 处理学历特征
        # 学历映射为数值（有序特征）
        education_mapping = {
            '高中': 1,
            '大专': 2,
            '本科': 3,
            '硕士': 4,
            '博士': 5,
            '学历不限': 0
        }

        # 保存映射关系
        self.education_mapping = education_mapping

        df['education_code'] = df['education_simple'].map(education_mapping)

        # 3. 处理工作经验特征
        # 将工作经验数值化
        X_experience = df['experience_years'].values.reshape(-1, 1)

        # 4. 组合特征
        X = np.column_stack([
            df['city_tier'].values,
            df['education_code'].values,
            df['experience_years'].values
        ])

        # 5. 目标变量（薪资）
        # 使用平均薪资作为预测目标
        y = df['salary_avg'].values

        print(f"特征形状: {X.shape}, 目标形状: {y.shape}")
        return X, y

    def build_model(self, input_dim):
        """构建深度学习模型"""
        print("构建深度学习模型...")

        model = keras.Sequential([
            # 输入层
            layers.Input(shape=(input_dim,)),

            # 隐藏层1
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # 隐藏层2
            layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            # 隐藏层3
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),

            # 输出层（预测薪资）
            layers.Dense(1, activation='linear')
        ])

        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # 均方误差
            metrics=['mae', 'mse']  # 平均绝对误差和均方误差
        )

        print("模型构建完成")
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        """训练模型"""
        print("开始训练模型...")

        # 早停机制
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        # 学习率调度
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )

        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )

        print("模型训练完成")
        return history

    def predict_salary(self, city, experience_years, education):
        """预测薪资"""
        # 准备输入特征
        city_tier = self.city_tier_map.get(city, self.city_tier_map['default'])
        education_code = self.education_mapping.get(education, 0)

        # 创建特征数组
        features = np.array([[city_tier, education_code, experience_years]])

        # 使用模型预测
        predicted_salary = self.model.predict(features, verbose=0)[0][0]

        # 根据城市等级和经验调整预测结果
        adjustment_factor = 1.0

        # 经验调整因子
        if experience_years >= 10:
            adjustment_factor *= 1.3
        elif experience_years >= 5:
            adjustment_factor *= 1.15
        elif experience_years >= 3:
            adjustment_factor *= 1.05

        # 学历调整因子
        if education_code >= 4:  # 硕士及以上
            adjustment_factor *= 1.2
        elif education_code == 3:  # 本科
            adjustment_factor *= 1.05

        # 应用调整因子
        adjusted_salary = predicted_salary * adjustment_factor

        # 生成薪资范围（±15%）
        salary_min = adjusted_salary * 0.85
        salary_max = adjusted_salary * 1.15

        return {
            'predicted_avg': round(adjusted_salary, 2),
            'salary_range': {
                'min': round(salary_min, 2),
                'max': round(salary_max, 2)
            },
            'city_tier': city_tier,
            'education_code': education_code,
            'adjustment_factor': round(adjustment_factor, 2)
        }

    def save_model(self, path='salary_predictor_model.h5'):
        """保存模型"""
        self.model.save(path)
        print(f"模型已保存到 {path}")

    def load_model(self, path='salary_predictor_model.h5'):
        """加载模型"""
        self.model = keras.models.load_model(path)
        print(f"模型已从 {path} 加载")


def main():
    """主函数"""
    print("薪资预测模型训练")

    # 初始化预测器
    predictor = SalaryPredictor()

    # 加载数据
    print("\n加载数据...")
    try:
        df = pd.read_csv('zhaopin_jobs.csv', encoding='utf-8')
        print(f"成功加载数据，共 {len(df)} 条记录")
    except FileNotFoundError:
        print("错误：未找到数据文件 'zhaopin_jobs.csv'")
        return

    # 3. 数据清洗
    print()
    df_clean = predictor.clean_data(df)

    # 4. 准备特征
    print()
    X, y = predictor.prepare_features(df_clean)

    # 5. 划分训练集和测试集
    print("\n划分训练集和测试集...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"训练集: {X_train.shape[0]} 条")
    print(f"验证集: {X_val.shape[0]} 条")
    print(f"测试集: {X_test.shape[0]} 条")

    # 6. 构建模型
    print()
    predictor.model = predictor.build_model(X_train.shape[1])

    # 7. 训练模型
    history = predictor.train(X_train, y_train, X_val, y_val, epochs=150)

    # 8. 评估模型
    test_loss, test_mae, test_mse = predictor.model.evaluate(X_test, y_test, verbose=0)
    print(f"测试集 MSE: {test_mse:.2f}")
    print(f"测试集 MAE: {test_mae:.2f} (约 {test_mae / 1000:.1f}K)")
    print(f"测试集 RMSE: {np.sqrt(test_mse):.2f}")

    # 9. 保存模型
    predictor.save_model()

    # 10. 示例预测
    print("\n示例预测:")

    # 使用固定的示例数据进行预测
    example_cases = [
        {'city': '北京', 'experience': 3.0, 'education': '本科'},
        {'city': '上海', 'experience': 5.0, 'education': '硕士'},
        {'city': '深圳', 'experience': 1.0, 'education': '大专'},
        {'city': '杭州', 'experience': 8.0, 'education': '本科'},
        {'city': '成都', 'experience': 2.0, 'education': '硕士'},
    ]

    
    for i, case in enumerate(example_cases, 1):
        try:
            result = predictor.predict_salary(
                city=case['city'],
                experience_years=case['experience'],
                education=case['education']
            )

            print(f"\n预测案例{i}")
            print(f"   城市: {case['city']}")
            print(f"   工作经验: {case['experience']} 年")
            print(f"   学历: {case['education']}")
            print(f"   城市等级系数: {result['city_tier']:.2f}")
            print(f"   调整因子: {result['adjustment_factor']:.2f}")
            print("-" * 30)
            print(f"   预测平均月薪: {result['predicted_avg']:,.0f} 元")
            print(f"   建议薪资范围: {result['salary_range']['min']:,.0f} - {result['salary_range']['max']:,.0f} 元")

            # 提供市场建议
            avg_salary = result['predicted_avg']
            if avg_salary < 8000:
                print("   - 市场建议: 该薪资处于初级水平，建议关注技能提升")
            elif avg_salary < 15000:
                print("   - 市场建议: 该薪资处于中等水平，具备市场竞争力")
            elif avg_salary < 25000:
                print("   - 市场建议: 该薪资处于中高水平，是资深人才的合理期望")
            else:
                print("   - 市场建议: 该薪资处于高水平，适合专家级人才")

        except Exception as e:
            print(f"   预测错误: {e}")


if __name__ == "__main__":
    main()