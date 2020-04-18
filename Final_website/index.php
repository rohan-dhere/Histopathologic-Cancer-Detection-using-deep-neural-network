<?php
session_start();

if(isset($_SESSION["loggedin_user"])){
	header("Location: upload_img.php");
  }

else{
	$_SESSION = array();
}  
?>



<!DOCTYPE html>
<html>

<head>
	<title>Login</title>
	<link rel="stylesheet" type="text/css" href="css/login.css">
	<link rel="stylesheet" type="text/css" href="css/main.css">
	<!--===============================================================================================-->


</head>

<body>
	<div class="container_login">
		<div class="wrap_login">
			<div class="login100-pic" data-tilt>
				<img src="images/144315.svg">
			</div>

			<form class="login100-form validate-form"   method="post" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>">

				<span class="login100-form-title">
					Patient Login
				</span>

				<div class="wrap-input100 validate-input">
					<input class="input100" type="text" name="email" placeholder="Email">

				</div>

				<div class="wrap-input100 validate-input">
					<input class="input100" type="password" name="pass" placeholder="Password">

				</div>

				<div class="container-login100-form-btn">
					<button class="login100-form-btn">
						Login
					</button>
				</div>

				<div class="text-center p-t-12">
					<span class="txt1">
						Forgot
					</span>
					<a class="txt2" href="forgot_password.html">
						Password?
					</a>
				</div>

				<div class="text-center p-t-136">
					<a class="txt2" href="create_ac.html">
						Create your Account
						<i class="fa fa-long-arrow-right m-l-5" aria-hidden="true"></i>
					</a>
				</div>
			</form>
		</div>
	</div>

</body>

</html>



<?php 

include 'php/config.php';

if ($_SERVER["REQUEST_METHOD"] == "POST") 
{

	$loggedin_user=$_POST['email'];
	$loggedin_passwrd=md5($_POST['pass']);
	$sql ="SELECT * FROM patients where email_id='$loggedin_user' ";
	$result=$conn->query($sql);
	while ($row=mysqli_fetch_assoc($result))
	 {
		$password=$row['password'];
		//$salt;
	}
	if($loggedin_passwrd == $password)
	{
		
		$_SESSION["loggedin_user"]=$loggedin_user;
		header("Location: upload_img.php");
		
	}
	else{
		echo "<script> alert('Sorry...Try Again'); </script> ";
	}
}

?>