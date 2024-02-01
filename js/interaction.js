function toggleForm(showFrom){
    var isRegister;
    if(showFrom == "register-form"){isRegister = true};
    var x = document.getElementsByClassName("register-form");
    x[0].style.display = isRegister ? "Block" : "None"; 
    
    var y = document.getElementsByClassName("login-form");
    y[0].style.display = isRegister ? "None" : "Block";
}

loginPlease.addEventListener('click', function(){
    toggleForm("login-form");
}, false);
registerPlease.addEventListener('click', function(){
    toggleForm("register-form");
}, false);

alert("1");