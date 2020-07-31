"use strict"
let xValue = [];
let yValue = [];

let m, b;

function setup() {
    createCanvas(800, 800);

    tf.tidy(() => {
        m = tf.variable(tf.scalar(random(1)));
        b = tf.variable(tf.scalar(random(1)));
    })
}

function mousePressed() {
    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, 0, height, 1, 0);

    xValue.push(x);
    yValue.push(y);
}

function predict(x) {
    const tfXs = tf.tensor1d(x);
    //Get y with mx plus b
    const tfYs = tfXs.mul(m).add(b);
    return tfYs;
}

function optimize(learningRate) {
    return tf.train.sgd(learningRate)
}

function loss(pred, labels) {
    return pred.sub(labels).square().mean();
}

function draw() {
    background(0);
    stroke(255);
    strokeWeight(4);

    //Train


    if (xValue.length > 0) {
        const optimizer = tf.train.sgd(.5);
        const ys = tf.tensor1d(yValue);
        tf.tidy(() => {
            optimizer.minimize(() => loss(predict(xValue), ys));
        })

        // draw aline
        const lineXs = [0, 1]
        const lineYs = tf.tidy(() => predict(lineXs));
        let yDataSync = lineYs.dataSync();
        lineYs.dispose();

        let x1 = map(lineXs[0], 0, 1, 0, width);
        let x2 = map(lineXs[1], 0, 1, 0, width);
        let y1 = map(yDataSync[0], 0, 1, height, 0);
        let y2 = map(yDataSync[1], 0, 1, height, 0);
        line(x1, y1, x2, y2)
    }

    for (let i = 0; i < xValue.length; i++) {
        let px = map(xValue[i], 0, 1, 0, width);
        let py = map(yValue[i], 0, 1, height, 0);

        point(px, py);
    }

    console.log(tf.memory().numTensors);
}