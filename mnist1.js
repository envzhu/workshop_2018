const X_SIZE = 28;
const Y_SIZE = 28;

const STEP_NUM = 150;
const MAX_TRAIN_NUM = 300;
const BATCH_SIZE = 8;
const EPOCHS = 1;
const REF_VAL = 0.8;

var train_files_0;
var train_files_1;
var test_files_0;
var test_files_1;

function load_data_1(f){
  return new Promise(resolve => {
    var reader = new FileReader();

    //ファイルの読込が終了した時の処理
    reader.onload = async function() {
      img = new Image();

      img.onload = async function(){
        var canvas = document.createElement('canvas');
        var context = canvas.getContext('2d');
        
        canvas.width = X_SIZE;
        canvas.height = Y_SIZE;
        context.drawImage(this, 0, 0, this.width, this.height, 0, 0, X_SIZE, Y_SIZE);
        
        let img_data = context.getImageData(0, 0, X_SIZE, Y_SIZE);
  
        let data = new Array(img_data.width * img_data.height);
  
        for (let j=0, k=0; j < img_data.width * img_data.height * 4; j+=4, k++) {
          //RGBのRだけを読み込む
          data[k] = img_data.data[j]/255;
        }
        resolve(data);
      };
      //読み込んだ画像ソースを入れる
      img.src = this.result;
     };

    //dataURL形式でファイルを読み込む
    reader.readAsDataURL(f);
  });
}

async function load_data(files){
  let data
  if(files.length>MAX_TRAIN_NUM)
    data = new Array(MAX_TRAIN_NUM);
  else 
    data = new Array(files.leng);

  for(let i=0; (i < files.length)&&(i<MAX_TRAIN_NUM); i++){
    console.log(files[i].name);
    data[i] = await load_data_1(files[i])
  }
  return data;
}

let clearHTML = function () {
  document.getElementById('console_log').innerHTML = "";
}

let putHTML = function (log) {
  console.log(log);
  document.getElementById('console_log').innerHTML += log + "<br>";
}

let statusHTML = function(status) {
  document.getElementById('status_log').innerHTML = status;
}

async function mnist1_main(){
    console.log("==========PROG2 START==========");
      // ライブラリのロード
    // name:visualization(可視化),version:バージョン(1),packages:パッケージ(corechart)
   // google.load('visualization', '1', {'packages':['corechart']});   
   STEP_NUM = document.form.learn_num.value;
   MAX_TRAIN_NUM = document.form.file_num.value;
   
    clearHTML();
    statusHTML("処理中");
    
    train_files_0 = document.getElementById('TrainFile_0').files;
    train_files_1 = document.getElementById('TrainFile_1').files;
    test_files_0 = document.getElementById('TestFile_0').files;
    test_files_1 = document.getElementById('TestFile_1').files;

    console.log(train_files_0);
    console.log(train_files_1);
    console.log(test_files_0);
    console.log(test_files_1);
    
    var train_data = new Array(2);
    var test_data = new Array(2);

    console.log("========== Load Train Data");
    train_data[0] = await load_data(train_files_0);
    train_data[1] = await load_data(train_files_1);

    /*
     * 学習用データを生成
     * 過学習を防ぐため、0と1を学習する順番は、
     * ランダムでなければならない
     */
    let tmp_x = new Array(train_data[0].length + train_data[1].length);
    let tmp_y = new Array(train_data[0].length + train_data[1].length);
    for(let i=0; i < train_data[0].length + train_data[1].length; i++){
      if(i%2 == 0){
        tmp_x[i] = train_data[0][i/2];
        tmp_y[i] = [1, 0];
      }else{
        tmp_x[i] = train_data[1][(i-1)/2];
        tmp_y[i] = [0, 1];
      }
    }

    console.log(tmp_x);
    console.log(tmp_y);

    let train_x = tf.tensor2d(tmp_x);
    let train_y = tf.tensor2d(tmp_y);

    delete tmp_x;
    delete tmp_y;
    delete train_data;

    let train_acc_log = new Array(STEP_NUM+1);
    for(let y = 0; y < STEP_NUM+1; y++) {
      train_acc_log[y] = new Array(3)
    }
    train_acc_log[0] = ["Epoch_num", "Train_loss", "Test_accuracy"];
    
    console.log("========== Load Test Data");
    test_data[0] = await load_data(test_files_0);
    console.log(test_data[0]);
    test_data[1] = await load_data(test_files_1);
    console.log(test_data[1]);


    console.log("========== Initial Model");

    const model = tf.sequential({
      layers: [tf.layers.dense({units: 2, inputShape: [X_SIZE*Y_SIZE]})]
    });
    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

    putHTML("First test accuracy "+ test_accuracy(model, test_data) + "%");

    console.log("========== Trainig");
    for(let i = 1; i <= STEP_NUM; i++){
      const history = await model.fit(
        train_x, train_y,
        { batchSize: BATCH_SIZE, epochs: EPOCHS });
        train_acc_log[i][0] = i;
        train_acc_log[i][1] = history.history.loss[0];
        train_acc_log[i][2] = test_accuracy(model, test_data);

      if((i%5 == 0)||(i == 1)){
        console.log("Loss after Epoch " + i + " : " + history.history.loss[0]);
        putHTML("Epoch :"+i + " Accuracy "+ test_accuracy(model, test_data) + "%");
      }
    }

    console.log("========== Test Accuracy");
    
    putHTML("Final test accuracy : " + test_accuracy(model, test_data, true) + "%");
    
    console.log(train_acc_log)
    // グラフの描画   
    google.charts.load('current', {packages: ['corechart', 'line']});
    google.charts.setOnLoadCallback(drawChart);
    function drawChart() {      
      
      // 配列からデータの生成
      var data = google.visualization.arrayToDataTable(train_acc_log);
    
      // オプションの設定
      var options = {
        title: '訓練データに対する正答率の推移'
       };     
               
      // 指定されたIDの要素に折れ線グラフを作成
      var chart = new google.visualization.LineChart(document.getElementById('chart_div'));
        
      // グラフの描画
      chart.draw(data, options);
    }

    statusHTML("処理終了");

    console.log("==============END==============");
}

function test_accuracy(model, data, display=false){
  let correct = 0;

  if(display)
    var p = console.log;
  else
    var p = function(){};

  p("1, 0");
  let output = model.predict(tf.tensor2d(data[0]));

  for(let i = 0; i < data[0].length*2;i+=2){
    if(output.dataSync()[i] > REF_VAL)correct++;
    p(`${output.dataSync()[i]}, ${output.dataSync()[i+1]}`);
  }

  p("0, 1");
  output = model.predict(tf.tensor2d(data[1]));

  for(let i=0; i<data[1].length*2;i+=2){
    if(output.dataSync()[i+1] > REF_VAL)correct++;
    p(`${output.dataSync()[i]}, ${output.dataSync()[i+1]}`);
  }

  return correct/(data[0].length+data[1].length)*100;

}
