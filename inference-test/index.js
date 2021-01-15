const axios = require("axios")
const logger = require("node-color-log")
const fs = require("fs");

const ROUND = 3;
const GPU_NUM = 3;
const BATCH_SIZE = 32; // Based on the JAVA batch size

async function send(deadline, layers, round, iter) {
    const res = await axios.put(
        `htp://127.0.0.1:8080/predictions/googlenet/${deadline}/${layers}`,
        fs.readFileSync(__dirname + '/kitten_small.jpg'), {
            headers: {
                'Content-Type': "jpg"
            }
        });
    logger.info(`Request in round ${round},${iter}:`, res.data)
}

async function sleep(ms) {
    await new Promise(resolve => setTimeout(resolve, ms));
    return;
}

function schedule(data) {
    var total_times = [
        [35.0641, 47.87375, 66.62346],
        [63.17956, 85.4668, 107.7465]
    ];
    var cut_times = [
        [7.93833, 10.75694, 13.56681, 20.18728, 24.05607],
        [11.75216, 14.69453, 17.85173, 21.06946, 25.92493, 31.57592, 39.26094, 46.94575],
        [11.75216, 25.92493, 31.57592, 39.26094, 46.94575]
    ];
    var cut_point = [//1->0, 2->0, 2->1
        [7, 8, 9, 11, 12],
        [7, 8, 9, 10, 11, 12, 13, 14],
        [7, 11, 12, 13, 14]
    ];
    var cut_total_times = [
        [41.73129, 44.20466, 46.05413, 46.23131, 46.62162],
        [46.06788, 48.84234, 51.04162, 52.42554, 52.43166, 54.60144, 58.60938, 63.07175],
        [56.38372, 57.99549, 59.77260, 62.53726, 65.31438]
    ];

    var cut = 0;
    var gpu_time = [0.0, 0.0, 0.0];
    var exec_seq = [0, 1, 2];
    for (let round = 0; round < ROUND; round++) {
        for(let gpu_num = 0; gpu_num < GPU_NUM; gpu_num++){
            if (gpu_time[exec_seq[0]] > gpu_time[exec_seq[1]]) {
                let temp = exec_seq[0];
                exec_seq[0] = exec_seq[1];
                exec_seq[1] = temp;
            }
            if (gpu_time[exec_seq[1]] > gpu_time[exec_seq[2]]) {
                let temp = exec_seq[1];
                exec_seq[1] = exec_seq[2];
                exec_seq[2] = temp;
            }
            if (gpu_time[exec_seq[0]] > gpu_time[exec_seq[1]]) {
                let temp = exec_seq[0];
                exec_seq[0] = exec_seq[1];
                exec_seq[1] = temp;
            }
            if (exec_seq[0] == 0){
                for (let i = 0; i < 32; i++) {
                    data[round * BATCH_SIZE * GPU_NUM + gpu_num * BATCH_SIZE + i].gpu_layers = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
                }
                gpu_time[0]+=total_times[BATCH_SIZE/32][0];
            }
            else if (exec_seq[0] == 1){
                let diff_time = gpu_time[0]-gpu_time[1];
                let idx = 0;
                for (idx = 0; idx < 5; idx++){
                    if (diff_time<cut_times[0][idx]){
                        cut = cut_point[0][idx];
                        gpu_time[0] = gpu_time[0] - diff_time + cut_total_times[0][idx];
                        gpu_time[1] += cut_times[0][idx];
                        break;
                    }
                }
                if (idx == 5){
                    for (let i = 0; i < 32; i++) {
                        data[round * BATCH_SIZE * GPU_NUM + gpu_num * BATCH_SIZE + i].gpu_layers = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1";
                    }
                    gpu_time[1]+=total_times[BATCH_SIZE/32][1];
                }else{
                    var gpu_layer = "";
                    for (let i = 0; i < 21; i++) {
                        if(i < cut + 1)
                            gpu_layer = gpu_layer.concat("1,");
                        else if(i < 20)
                            gpu_layer = gpu_layer.concat("0,");
                        else
                            gpu_layer = gpu_layer.concat("0");
                    }
                    for (let i = 0; i < 32; i++) {
                        data[round * BATCH_SIZE * GPU_NUM + gpu_num * BATCH_SIZE + i].gpu_layers = gpu_layer.slice();
                    }
                }
            }else{
                let diff_time = gpu_time[0]-gpu_time[2];
                let diff_time2 = gpu_time[1]-gpu_time[2];
                let choice = 1;
                let max_idx = 7;
                if (diff_time > 31.57592 && diff_time2 < 25.92493){
                    choice = 2;
                    max_idx = 5;
                    diff_time = diff_time2;
                }else if(diff_time > 39.26094 && diff_time2 < 39.26094){
                    choice = 2;
                    max_idx = 5;
                    diff_time = diff_time2;
                }
                let idx = 0;
                for (idx = 0; idx < max_idx; idx++){
                    if (diff_time<cut_times[choice][idx]){
                        cut = cut_point[choice][idx];
                        gpu_time[choice-1] = gpu_time[choice-1] - diff_time + cut_total_times[choice][idx];
                        gpu_time[2] += cut_times[choice][idx];
                        break;
                    }
                }
                if (idx == max_idx){
                    for (let i = 0; i < 32; i++) {
                        data[round * BATCH_SIZE * GPU_NUM + gpu_num * BATCH_SIZE + i].gpu_layers = "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2";
                    }
                    gpu_time[2]+=total_times[BATCH_SIZE/32][2];
                }else{
                    var gpu_layer = "";
                    for (let i = 0; i < 21; i++) {
                        if(i < cut + 1)
                            gpu_layer = gpu_layer.concat("2,");
                        else if(i < 20)
                            gpu_layer = gpu_layer.concat((choice-1)+",");
                        else
                            gpu_layer = gpu_layer.concat((choice-1));
                    }
                    for (let i = 0; i < 32; i++) {
                        data[round * BATCH_SIZE * GPU_NUM + gpu_num * BATCH_SIZE + i].gpu_layers = gpu_layer.slice();;
                    }
                }
            }
        }
    }
}

function gen_data() {
    const data = []


    for (let round = 0; round < ROUND; round++) {
        for (let i = 0; i < BATCH_SIZE * GPU_NUM; i++) {
            data.push({
                deadline: 100000000,
                gpu_layers: undefined,
            });
        }
    }

    return data;
}

async function send_all(data) {

    for (let round = 0; round < ROUND; round++) {
        for (let i = 0; i < BATCH_SIZE * GPU_NUM; i++) {
            const d = data[round * BATCH_SIZE * GPU_NUM + i];
            send(d.deadline, d.gpu_layers, round, i)
        }

        // change the time to modify the frequency of sending requests
        await sleep(50); // 50ms
    }
}

async function main() {
    const data = gen_data();
    schedule(data);
    send_all(data);
}

main();
