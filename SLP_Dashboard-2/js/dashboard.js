/*const studentData = {
    student1: { 
        accuracyTime: [70, 75, 80, 85], 
        letterCombination: { S: [90, 85, 70, 95], R: [80, 70, 75, 90] }, 
        gameLevels: [60, 75, 80, 95], 
        confidence: [50, 30, 20],
        syllables: [60, 70, 80, 90],
        soundLocation: [75, 85, 90],
        minutesPracticed: [30, 45, 50, 60]  // New data
    },
    student2: { 
        accuracyTime: [60, 70, 75, 80], 
        letterCombination: { S: [85, 80, 75, 90], R: [70, 65, 60, 85] }, 
        gameLevels: [55, 65, 75, 85], 
        confidence: [45, 35, 20],
        syllables: [55, 65, 75, 85],
        soundLocation: [65, 75, 80],
        minutesPracticed: [25, 35, 40, 55]  // New data
    },
    student3: { 
        accuracyTime: [65, 72, 78, 82], 
        letterCombination: { S: [88, 77, 66, 99], R: [75, 85, 70, 80] }, 
        gameLevels: [45, 55, 65, 75], 
        confidence: [40, 40, 20],
        syllables: [50, 60, 70, 80],
        soundLocation: [70, 80, 85],
        minutesPracticed: [20, 30, 45, 50]  // New data
    },
    student4: { 
        accuracyTime: [21, 32, 14, 12], 
        letterCombination: { S: [12, 10, 5, 7], R: [9, 16, 2, 10] }, 
        gameLevels: [45, 55, 65, 75], 
        confidence: [0, 20, 80],
        syllables: [50, 60, 70, 80],
        soundLocation: [70, 80, 85],
        minutesPracticed: [20, 30, 45, 50]  // New data
    }
};

// Initialize charts as empty variables so we can update them later
let accuracyTimeChart, letterCombinationChart, gameLevelChart, confidenceChart;
let accuracyBySyllablesChart, accuracyBySoundLocationChart, minutesPracticedChart;

// Default sound
let selectedSound = 'S';

// Function to initialize charts with the first student's data
function loadCharts() {
    const data = studentData.student1;

    // Accuracy Time Chart (existing code)
    const accuracyTimeCtx = document.getElementById("accuracyTimeChart");
    if (accuracyTimeCtx) {
        accuracyTimeChart = new Chart(accuracyTimeCtx, {
            type: 'line',
            data: {
                labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                datasets: [{
                    label: 'Accuracy Over Time',
                    data: data.accuracyTime,
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }
        });
    }

    // Letter Combination Chart
    const letterCombinationCtx = document.getElementById("letterCombinationChart");
    if (letterCombinationCtx) {
        letterCombinationChart = new Chart(letterCombinationCtx, {
            type: 'pie',
            data: {
                labels: [`${selectedSound}a`, `${selectedSound}e`, `${selectedSound}i`, `${selectedSound}o`],
                datasets: [{
                    label: 'Accuracy by Letter Combinations',
                    data: data.letterCombination[selectedSound],
                    backgroundColor: ['rgba(255, 99, 132, 0.2)', 'rgba(54, 162, 235, 0.2)', 'rgba(255, 206, 86, 0.2)', 'rgba(75, 192, 192, 0.2)'],
                    borderColor: ['rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)', 'rgba(255, 206, 86, 1)', 'rgba(75, 192, 192, 1)'],
                    borderWidth: 1
                }]
            },
            options: { responsive: true }
        });
    }

    // Game Level Chart
    const gameLevelCtx = document.getElementById("gameLevelChart");
    if (gameLevelCtx) {
        gameLevelChart = new Chart(gameLevelCtx, {
            type: 'bar',
            data: {
                labels: ['Level 1', 'Level 2', 'Level 3', 'Level 4'],
                datasets: [{
                    label: 'Progress by Game Level',
                    data: data.gameLevels,
                    backgroundColor: 'rgba(153, 102, 255, 0.2)',
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderWidth: 1
                }]
            },
            options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }
        });
    }

    // Confidence Chart
    const confidenceCtx = document.getElementById("confidenceChart");
    if (confidenceCtx) {
        confidenceChart = new Chart(confidenceCtx, {
            type: 'doughnut',
            data: {
                labels: ['Correct', 'Unsure', 'Incorrect'],
                datasets: [{
                    label: 'Pronunciation Confidence Level',
                    data: data.confidence,
                    backgroundColor: ['rgba(75, 192, 192, 0.2)', 'rgba(255, 159, 64, 0.2)', 'rgba(255, 99, 132, 0.2)'],
                    borderColor: ['rgba(75, 192, 192, 1)', 'rgba(255, 159, 64, 1)', 'rgba(255, 99, 132, 1)'],
                    borderWidth: 1
                }]
            },
            options: { responsive: true }
        });
    }

    // Accuracy by Syllables Chart
    const syllablesCtx = document.getElementById("syllablesChart");
    if (syllablesCtx) {
        accuracyBySyllablesChart = new Chart(syllablesCtx, {
            type: 'line',
            data: {
                labels: ['Syllable 1', 'Syllable 2', 'Syllable 3', 'Syllable 4'],
                datasets: [{
                    label: 'Accuracy by Syllables',
                    data: data.syllables,
                    backgroundColor: 'rgba(255, 205, 86, 0.2)',
                    borderColor: 'rgba(255, 205, 86, 1)',
                    borderWidth: 1
                }]
            },
            options: { responsive: true }
        });
    }

    // Accuracy by Sound Location Chart
    const soundLocationCtx = document.getElementById("soundLocationChart");
    if (soundLocationCtx) {
        accuracyBySoundLocationChart = new Chart(soundLocationCtx, {
            type: 'radar',
            data: {
                labels: ['Beginning', 'Middle', 'End'],
                datasets: [{
                    label: 'Accuracy by Sound Location',
                    data: data.soundLocation,
                    backgroundColor: 'rgba(201, 203, 207, 0.2)',
                    borderColor: 'rgba(201, 203, 207, 1)',
                    borderWidth: 1
                }]
            },
            options: { responsive: true, scales: { r: { beginAtZero: true, max: 100 } } }
        });
        const minutesPracticedCtx = document.getElementById("minutesPracticedChart");
        if (minutesPracticedCtx) {
            minutesPracticedChart = new Chart(minutesPracticedCtx, {
                type: 'bar',
                data: {
                    labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                    datasets: [{
                        label: 'Minutes Practiced',
                        data: data.minutesPracticed,
                        backgroundColor: 'rgba(255, 159, 64, 0.2)',
                        borderColor: 'rgba(255, 159, 64, 1)',
                        borderWidth: 1
                    }]
                },
                options: { responsive: true, scales: { y: { beginAtZero: true } } }
            });
        }
    }
}

function selectStudent(studentName) {
    if (!studentName || !studentData[studentName]) return;

    const data = studentData[studentName];

    // Update Accuracy Over Time chart
    if (accuracyTimeChart) {
        accuracyTimeChart.data.datasets[0].data = data.accuracyTime;
        accuracyTimeChart.update();
    }

    // Update Letter Combination chart
    if (letterCombinationChart) {
        letterCombinationChart.data.datasets[0].data = data.letterCombination[selectedSound];
        letterCombinationChart.update();
    }

    // Update Game Level chart
    if (gameLevelChart) {
        gameLevelChart.data.datasets[0].data = data.gameLevels;
        gameLevelChart.update();
    }

    // Update Pronunciation Confidence chart
    if (confidenceChart) {
        confidenceChart.data.datasets[0].data = data.confidence;
        confidenceChart.update();
    }

    // Update Accuracy by Syllables chart
    if (accuracyBySyllablesChart) {
        accuracyBySyllablesChart.data.datasets[0].data = data.syllables;
        accuracyBySyllablesChart.update();
    }

    // Update Accuracy by Sound Location chart
    if (accuracyBySoundLocationChart) {
        accuracyBySoundLocationChart.data.datasets[0].data = data.soundLocation;
        accuracyBySoundLocationChart.update();
    }

    // Update Minutes Practiced chart
    if (minutesPracticedChart) {
        minutesPracticedChart.data.datasets[0].data = data.minutesPracticed;
        minutesPracticedChart.update();
    }
}

// Call loadCharts() initially to load the first student's data on page load
document.addEventListener("DOMContentLoaded", loadCharts);

// Call loadCharts on page load
window.onload = loadCharts;
*/