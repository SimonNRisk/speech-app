/*// Object to store notes for each student by week
const studentNotes = {};

function toggleNotes() {
    const notesSection = document.getElementById("notesSection");
    const chartGrid = document.querySelector(".chart-grid");
    
    if (notesSection.style.display === "none") {
        notesSection.style.display = "flex"; // Change to flex to apply full height
        chartGrid.style.display = "none"; // Hide charts
    } else {
        notesSection.style.display = "none";
        chartGrid.style.display = "grid"; // Show charts
    }
}

function loadNotesForWeek(studentId, weekId) {
    if (!studentNotes[studentId]) studentNotes[studentId] = {};
    document.getElementById("notesTextArea").value = studentNotes[studentId][weekId] || "";
}

function saveNotesForWeek(studentId, weekId, notes) {
    if (!studentNotes[studentId]) studentNotes[studentId] = {};
    studentNotes[studentId][weekId] = notes;
}

function selectWeekTab(weekId) {
    const selectedStudent = document.getElementById("studentDropdown").value;
    if (!selectedStudent) return;

    const currentTab = document.querySelector(".tab.active");
    if (currentTab) {
        const currentWeekId = currentTab.getAttribute("data-week");
        const currentNotes = document.getElementById("notesTextArea").value;
        saveNotesForWeek(selectedStudent, currentWeekId, currentNotes);
    }

    document.querySelectorAll(".tab").forEach(tab => tab.classList.remove("active"));
    document.querySelector(`.tab[data-week="${weekId}"]`).classList.add("active");
    loadNotesForWeek(selectedStudent, weekId);
}

function addNewWeek() {
    const weekTabs = document.getElementById("weekTabs");
    const newWeekNumber = weekTabs.children.length + 1;
    const newWeekId = `week${newWeekNumber}`;

    const newTab = document.createElement("span");
    newTab.className = "tab";
    newTab.textContent = `Week ${newWeekNumber}`;
    newTab.setAttribute("data-week", newWeekId);
    newTab.classList.add("tab");

    weekTabs.appendChild(newTab);
    selectWeekTab(newWeekId);
}

document.getElementById("weekTabs").addEventListener("click", function(event) {
    if (event.target.classList.contains("tab")) {
        const weekId = event.target.getAttribute("data-week");
        selectWeekTab(weekId);
    }
});

document.getElementById("notesButton").addEventListener("click", function() {
    document.getElementById("chartsContainer").style.display = "none";
    document.getElementById("notesSection").style.display = "block";

    const selectedStudent = document.getElementById("studentDropdown").value;
    if (selectedStudent) {
        selectWeekTab("week1");
    }
});

function selectStudent(studentId) {
    if (!studentId) return;

    const studentName = studentId.charAt(0).toUpperCase() + studentId.slice(1);
    document.getElementById("dashboardTitle").textContent = `SLP Dashboard - ${studentName}`;

    document.getElementById("chartsContainer").style.display = "block";
    document.getElementById("notesSection").style.display = "none";
}

document.addEventListener("DOMContentLoaded", function () {
    if (document.querySelector(".tab")) {
        selectWeekTab("week1");
    }
});
*/