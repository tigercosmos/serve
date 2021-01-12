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
    for (let round = 0; round < ROUND; round++) {
        // GPU 0
        for (let i = 0; i < BATCH_SIZE; i++) {
            data[round * BATCH_SIZE * GPU_NUM + i].gpu_layers = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
        }

        // GPU 1
        for (let i = BATCH_SIZE; i < BATCH_SIZE * 2; i++) {
            data[round * BATCH_SIZE * GPU_NUM + i].gpu_layers = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
        }

        // GPU 2
        for (let i = BATCH_SIZE * 2; i < BATCH_SIZE * 3; i++) {
            data[round * BATCH_SIZE * GPU_NUM + i].gpu_layers = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
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