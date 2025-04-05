/* menu-button */

document.addEventListener("DOMContentLoaded", () => {
    const menuButton=document.querySelector('.menu-button');
    const mobileSidebar=document.getElementById('mobileSidebar');
    const closeSidebar = document.getElementById('closeSidebar');

    menuButton.addEventListener('click', () => {
        mobileSidebar.classList.add('visible');
        mobileSidebar.classList.remove('hidden');
    });

    closeSidebar.addEventListener('click', () => {
        mobileSidebar.classList.remove('visible');
        setTimeout(() => {
            mobileSidebar.classList.add('hidden');
        }, 300);
    }) 

})