use nalgebra::{DMatrix, DVector};
use csv::Reader;
use std::error::Error;
use rand_distr::{Normal, Distribution};
use std::collections::HashMap;
use rand::seq::SliceRandom;
use actix_web::{web, App, HttpServer, HttpResponse, middleware};
use actix_cors::Cors;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

#[derive(Clone)]
struct LogisticRegression {
    weights: DVector<f64>,
    learning_rate: f64,
    weight_history: Vec<DVector<f64>>,
}

impl LogisticRegression {
    fn new(number_of_array: usize, learning_rate: f64) -> Self {
        LogisticRegression {
            weights: DVector::zeros(number_of_array),
            learning_rate,
            weight_history: Vec::new(),
        }
    }
    
    fn sigmoid(val: f64) -> f64 { 
        1.0 / (1.0 + (-val).exp()) 
    }

    fn standardize(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut feature = x.clone(); 
        for col in 0..feature.ncols() {
            let col_vec = feature.column(col);
            let mean = col_vec.iter().sum::<f64>() / col_vec.len() as f64;
            let std = (col_vec.iter().map(|&value_x_row_y_column| 
                 (value_x_row_y_column - mean).powi(2)).sum::<f64>() / col_vec.len() as f64).sqrt();
            if std > 1e-8 { // 避免除零
                for i in 0..feature.nrows(){
                    feature[(i,col)]=(feature[(i,col)]-mean)/std;
                }
            }
        } 
        feature
    }

    fn gradient_calculation(&self, feature_matrix: &DMatrix<f64>, label: &DVector<f64>) -> DVector<f64> {
        let linear_layer = feature_matrix * &self.weights;
        let activate_layer: DVector<f64> = linear_layer.map(Self::sigmoid);
        let feature_row = feature_matrix.nrows() as f64;
        feature_matrix.transpose() * (&activate_layer - label) / feature_row
    }

    fn calculate_loss(&self, features: &DMatrix<f64>, labels: &DVector<f64>) -> f64 {
        let z = features * &self.weights;
        let sigmoid_z: DVector<f64> = z.map(Self::sigmoid);
        let m = features.nrows() as f64;
        
        -labels.iter()
            .zip(sigmoid_z.iter())
            .map(|(y_true, y_pred)| {
                let y_true_f64 = *y_true;
                let y_pred_f64 = y_pred.max(1e-15).min(1.0 - 1e-15); // 防止log(0)
                y_true_f64 * y_pred_f64.ln() + (1.0 - y_true_f64) * (1.0 - y_pred_f64).ln()
            })
            .sum::<f64>() / m
    }

    fn get_batch(
        features: &DMatrix<f64>, 
        labels: &DVector<f64>, 
        batch_idx: usize, 
        batch_size: usize
    ) -> (DMatrix<f64>, DVector<f64>) {
        let n_samples = features.nrows();
        let start = batch_idx * batch_size;
        let end = std::cmp::min(start + batch_size, n_samples);
        let batch_size_actual = end - start;
        
        let mut batch_features = DMatrix::zeros(batch_size_actual, features.ncols());
        let mut batch_labels = DVector::zeros(batch_size_actual);
        
        for i in 0..batch_size_actual {
            for j in 0..features.ncols() {
                batch_features[(i, j)] = features[(start + i, j)];
            }
            batch_labels[i] = labels[start + i];
        }
        
        (batch_features, batch_labels)
    }

    fn train(&mut self, feature: &DMatrix<f64>, label: &DVector<f64>, iterations: usize, batch_size: usize) -> Result<Vec<f64>, Box<dyn Error>> {
        let standardized_feature = self.standardize(feature);
        let mut loss_history = Vec::with_capacity(iterations);
        self.weight_history.clear();
        self.weight_history.push(self.weights.clone());
        
        let n_samples = standardized_feature.nrows();
        let n_batches = (n_samples + batch_size - 1) / batch_size;
        
        for _epoch in 0..iterations {
            let mut epoch_loss = 0.0;
            
            for batch_idx in 0..n_batches {
                let (batch_features, batch_labels) = Self::get_batch(&standardized_feature, label, batch_idx, batch_size);
                
                let gradient = self.gradient_calculation(&batch_features, &batch_labels);
                self.weights -= &gradient.scale(self.learning_rate);
                
                self.weight_history.push(self.weights.clone());
                
                epoch_loss += self.calculate_loss(&batch_features, &batch_labels) * batch_features.nrows() as f64;
            }
            
            epoch_loss /= n_samples as f64;
            loss_history.push(epoch_loss);
        }
        
        Ok(loss_history)
    }

    fn predict(&self, feature: &DMatrix<f64>) -> DVector<f64> {
        let standardized_feature = self.standardize(feature);
        let z = &standardized_feature * &self.weights;
        z.map(Self::sigmoid).map(|p| if p >= 0.5 { 1.0 } else { 0.0 })
    }
}

#[derive(Serialize)]
struct AgeGroupAnalysis {
    age_group: String,
    sample_count: usize,
    actual_survival_rate: f64,
    predicted_survival_rate: f64,
    difference_rate: f64,
}

fn analyze_predictions_by_age_group(
    x: &DMatrix<f64>, 
    y: &DVector<f64>, 
    predictions: &DVector<f64>, 
    age_group_size: usize,
    age_col_index: Option<usize>
) -> Vec<AgeGroupAnalysis> {
    if let Some(age_idx) = age_col_index {
        let ages: Vec<f64> = (0..x.nrows()).map(|i| x[(i, age_idx)]).collect();
        
        let mut age_groups: HashMap<usize, (usize, usize, usize)> = HashMap::new();
        
        for i in 0..ages.len() {
            let age = ages[i] as usize;
            let group = (age / age_group_size) * age_group_size;
            
            let actual = y[i] as usize;
            let predicted = predictions[i] as usize;
            
            let entry = age_groups.entry(group).or_insert((0, 0, 0));
            entry.0 += 1;
            entry.1 += actual;
            entry.2 += predicted;
        }
        
        let mut groups: Vec<(usize, (usize, usize, usize))> = 
            age_groups.into_iter().collect();
        groups.sort_by_key(|&(group, _)| group);
        
        let mut results = Vec::new();
        
        for &(group, (total, actual_survived, predicted_survived)) in &groups {
            let actual_rate = if total > 0 { actual_survived as f64 / total as f64 * 100.0 } else { 0.0 };
            let predicted_rate = if total > 0 { predicted_survived as f64 / total as f64 * 100.0 } else { 0.0 };
            let diff_rate = predicted_rate - actual_rate;
            
            results.push(AgeGroupAnalysis {
                age_group: format!("{}-{}", group, group + age_group_size - 1),
                sample_count: total,
                actual_survival_rate: actual_rate,
                predicted_survival_rate: predicted_rate,
                difference_rate: diff_rate,
            });
        }
        
        results
    } else {
        Vec::new() // 如果没有年龄列，返回空的分析
    }
}

fn load_data_auto(file_path: &str, target_column: &str) -> Result<(DMatrix<f64>, DVector<f64>, Option<usize>, Vec<String>), Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;
    
    // 读取表头
    let headers = reader.headers()?.clone();
    let header_names: Vec<String> = headers.iter().map(|h| h.to_string()).collect();
    
    // 找到目标列的索引
    let target_col_idx = headers.iter()
        .position(|h| h.eq_ignore_ascii_case(target_column))
        .ok_or_else(|| format!("未找到目标列: {}", target_column))?;
    
    // 收集所有记录
    let mut records = Vec::new();
    for result in reader.records() {
        records.push(result?);
    }
    
    if records.is_empty() {
        return Err("CSV文件为空".into());
    }
    
    // 分析哪些列是数值列（排除目标列）
    let mut numeric_columns = Vec::new();
    let mut age_col_index: Option<usize> = None;
    
    for (col_idx, header) in headers.iter().enumerate() {
        if col_idx == target_col_idx {
            continue; // 跳过目标列
        }
        
        // 检查这一列是否大部分都是数值
        let mut numeric_count = 0;
        let mut total_count = 0;
        
        for record in &records {
            if let Some(value) = record.get(col_idx) {
                total_count += 1;
                if !value.trim().is_empty() && value.parse::<f64>().is_ok() {
                    numeric_count += 1;
                }
            }
        }
        
        // 如果至少70%的值是数值，则认为是数值列
        if total_count > 0 && (numeric_count as f64 / total_count as f64) >= 0.7 {
            numeric_columns.push(col_idx);
            
            // 检查是否是年龄列
            if header.to_lowercase().contains("age") {
                age_col_index = Some(numeric_columns.len() - 1);
            }
        }
    }
    
    if numeric_columns.is_empty() {
        return Err("未找到任何数值特征列".into());
    }
    
    // 构建特征列名
    let feature_names: Vec<String> = numeric_columns.iter()
        .map(|&idx| header_names[idx].clone())
        .collect();
    
    // 为缺失的数值计算均值（用于填充）
    let mut column_means = Vec::new();
    for &col_idx in &numeric_columns {
        let mut sum = 0.0;
        let mut count = 0;
        
        for record in &records {
            if let Some(value) = record.get(col_idx) {
                if let Ok(num) = value.trim().parse::<f64>() {
                    sum += num;
                    count += 1;
                }
            }
        }
        
        column_means.push(if count > 0 { sum / count as f64 } else { 0.0 });
    }
    
    // 构建特征矩阵和标签向量
    let mut features_data = Vec::new();
    let mut labels_data = Vec::new();
    
    for record in &records {
        // 获取目标值
        if let Some(target_str) = record.get(target_col_idx) {
            if let Ok(target_val) = target_str.trim().parse::<f64>() {
                labels_data.push(target_val);
                
                // 获取特征值
                let mut feature_row = Vec::new();
                for (i, &col_idx) in numeric_columns.iter().enumerate() {
                    if let Some(value_str) = record.get(col_idx) {
                        let value = value_str.trim().parse::<f64>()
                            .unwrap_or(column_means[i]); // 使用均值填充缺失值
                        feature_row.push(value);
                    } else {
                        feature_row.push(column_means[i]);
                    }
                }
                features_data.push(feature_row);
            }
        }
    }
    
    if features_data.is_empty() {
        return Err("没有有效的数据行".into());
    }
    
    // 转换为矩阵
    let feature_cols: Vec<DVector<f64>> = (0..numeric_columns.len())
        .map(|col_idx| {
            DVector::from_vec(
                features_data.iter().map(|row| row[col_idx]).collect()
            )
        })
        .collect();
    
    let x = DMatrix::from_columns(&feature_cols);
    let y = DVector::from_vec(labels_data);
    
    Ok((x, y, age_col_index, feature_names))
}

fn split(
    x: &DMatrix<f64>, 
    y: &DVector<f64>,
    test_size: f64,
) -> Result<(DMatrix<f64>, DMatrix<f64>, DVector<f64>, DVector<f64>), Box<dyn Error>> {
    let row = x.nrows();
    let n_test = (row as f64 * test_size).round() as usize;
    let n_train = row - n_test;
    
    let mut indices: Vec<usize> = (0..row).collect();
    indices.shuffle(&mut rand::thread_rng());
    
    let mut x_train = DMatrix::zeros(n_train, x.ncols());
    let mut y_train = DVector::zeros(n_train);
    let mut x_test = DMatrix::zeros(n_test, x.ncols());
    let mut y_test = DVector::zeros(n_test);
    
    for (i, &idx) in indices.iter().take(n_train).enumerate() {
        for j in 0..x.ncols() {
            x_train[(i, j)] = x[(idx, j)];
        }
        y_train[i] = y[idx];
    }
    
    for (i, &idx) in indices.iter().skip(n_train).enumerate() {
        for j in 0..x.ncols() {
            x_test[(i, j)] = x[(idx, j)];
        }
        y_test[i] = y[idx];
    }
    
    Ok((x_train, x_test, y_train, y_test))
}

#[derive(Deserialize)]
struct TrainRequest {
    file_path: String,
    target_column: String,
    learning_rate: f64,
    batch_size: usize,
    epochs: usize,
    test_ratio: f64,
}

#[derive(Serialize)]
struct TrainResponse {
    train_accuracy: f64,
    test_accuracy: f64,
    age_group_analysis: Vec<AgeGroupAnalysis>,
    weight_history: Vec<Vec<f64>>,
    loss_history: Vec<f64>,
    feature_names: Vec<String>,
    total_samples: usize,
    train_samples: usize,
    test_samples: usize,
    numeric_features: usize,
    test_features: Vec<Vec<f64>>,
    test_labels: Vec<f64>,
    test_predictions: Vec<f64>,
}

async fn train_model(
    data: web::Json<TrainRequest>
) -> HttpResponse {
    match load_data_auto(&data.file_path, &data.target_column) {
        Ok((x, y, age_col_index, feature_names)) => {
            let test_size = data.test_ratio;
            let (x_train, x_test, y_train, y_test) = match split(&x, &y, test_size) {
                Ok(result) => result,
                Err(e) => return HttpResponse::InternalServerError().json(format!("数据分割错误: {}", e))
            };
            
            let mut model = LogisticRegression::new(x.ncols(), data.learning_rate);
            
            let loss_history = match model.train(&x_train, &y_train, data.epochs, data.batch_size) {
                Ok(history) => history,
                Err(e) => return HttpResponse::InternalServerError().json(format!("模型训练错误: {}", e))
            };
            
            let train_predictions = model.predict(&x_train);
            let test_predictions = model.predict(&x_test);
            
            let train_accuracy = train_predictions.iter()
                .zip(y_train.iter())
                .filter(|&(pred, true_label)| (pred - true_label).abs() < 1e-10)
                .count() as f64 / y_train.len() as f64;
            
            let test_accuracy = test_predictions.iter()
                .zip(y_test.iter())
                .filter(|&(pred, true_label)| (pred - true_label).abs() < 1e-10)
                .count() as f64 / y_test.len() as f64;
            
            let age_analysis = analyze_predictions_by_age_group(&x_test, &y_test, &test_predictions, 5, age_col_index);
            
            let weight_history: Vec<Vec<f64>> = model.weight_history.iter()
                .map(|w| w.iter().copied().collect())
                .collect();
            
            // 准备测试集数据用于前端分析
            let test_features_vec: Vec<Vec<f64>> = (0..x_test.ncols())
                .map(|col| (0..x_test.nrows()).map(|row| x_test[(row, col)]).collect())
                .collect();
            
            let test_labels_vec: Vec<f64> = y_test.iter().copied().collect();
            let test_predictions_vec: Vec<f64> = test_predictions.iter().copied().collect();
            
            HttpResponse::Ok().json(TrainResponse {
                train_accuracy,
                test_accuracy,
                age_group_analysis: age_analysis,
                weight_history,
                loss_history,
                feature_names,
                total_samples: x.nrows(),
                train_samples: x_train.nrows(),
                test_samples: x_test.nrows(),
                numeric_features: x.ncols(),
                test_features: test_features_vec,
                test_labels: test_labels_vec,
                test_predictions: test_predictions_vec,
            })
        }
        Err(e) => HttpResponse::InternalServerError().json(format!("数据加载错误: {}", e))
    }
}

async fn index() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/html")
        .body(include_str!("index.html"))
}

async fn analysis() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/html")
        .body(include_str!("analysis.html"))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Server running at http://localhost:8080");
    
    HttpServer::new(|| {
        App::new()
            .wrap(
                Cors::default()
                    .allow_any_origin()
                    .allow_any_method()
                    .allow_any_header()
            )
            .wrap(middleware::Logger::default())
            .route("/", web::get().to(index))
            .route("/train", web::post().to(train_model))
            .route("/analysis", web::get().to(analysis))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}