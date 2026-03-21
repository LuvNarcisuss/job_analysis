# main.py
import os
from data_spider import ZhaopinSpider
from analysis import Analyzer
from visualization import DataVisualization

def main():
    """ 主运行程序 """
    print("=" * 60)
    print("职位招聘市场分析系统")
    print("=" * 60)

    # 1. 数据爬取

    # 检查是否存在数据文件
    if not os.path.exists('zhaopin_jobs.csv'):
        print("未找到数据文件，开始执行数据爬取...")
        try:
            print("\n开始执行数据爬取...")
            spider = ZhaopinSpider('zhaopin_jobs.csv')
            keywords = ['python', 'java', 'go', 'c++', 'javascript']
            spider.run(keywords, 50)
            print("数据爬取完成")
        except Exception as e:
            print(f"数据爬取失败: {e}")
            return
    else:
        print("数据文件已存在，跳过爬取步骤")

    # 2. 数据分析

    try:
        print("\n开始执行数据分析...")
        analyzer = Analyzer('zhaopin_jobs.csv')
        analyzer.load_data()
        analyzer.data_preprocessing()
        analyzer.generate_statistical_report()
        model_results = analyzer.build_prediction_model()
        analyzer.generate_final_report()
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
        print("\n数据分析完成")
    except Exception as e:
        print(f"数据分析失败: {e}")
        return

    # 3. 数据可视化

    try:
        print("\n开始执行数据可视化...")
        # 使用分析模块处理过的数据
        viz = DataVisualization(analyzer.df)
        viz.create_all_basic_visualizations()
        viz.create_all_advanced_visualizations()

        # 如果有模型结果，也生成相关图表
        if model_results:
            viz.create_feature_importance_plot(model_results['feature_importance'])
        print("数据可视化完成")
    except Exception as e:
        print(f"数据可视化失败: {e}")
        return

    print("\n" + "=" * 60)
    print("所有模块运行完成！")
    print("邓双林666")
    print("请查看生成的图表文件和分析报告")
    print("=" * 60)

if __name__ == "__main__":
    main()
