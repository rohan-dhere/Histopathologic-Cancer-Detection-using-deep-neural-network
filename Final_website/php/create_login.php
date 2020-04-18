<?php
$dbservername = "localhost";
$dbusername = "root";
$dbpassword = "123456";
$dbname="cancer_detection";
$conn = mysqli_connect($dbservername, $dbusername, $dbpassword,$dbname);


if ( !$conn ) {
    echo ( 'Did not connect: ' . mysqli_connect_error() ); 
}


$salt=md5(rand(10,10000));


$contact = $_POST['contact'];
$cell_image = "";

$name=$_POST['name'];
$email=$_POST['email'];
$id = substr(md5($email),0,5);
$age=$_POST['age'];
$sex=$_POST['sex'];
$password=md5($_POST['pass']);
$password_confirm=md5($_POST['pass_confirm']);
$contact = $_POST['contact'];
$date_of_birth=$_POST['date_of_birth'];
$blood_group=$_POST['blood_group'];
$address=$_POST['address'];
$medical_history=$_POST['medical_history'];




$sql = "INSERT INTO `patients` (`id`, `password`, `salt`, `name`, `age`, `date of birth`, `sex`, `blood_group`, `contact`, `email_id`, `address`, `medical_history`, `cell_image`) VALUES ('$id', '$password', '$salt', '$name', '$age', '$date_of_birth', '$sex', '$blood_group', '$contact', '$email','$address', '$medical_history', '$cell_image')";

if($password != $password_confirm)
{
	echo "<script> alert('Passwords not matching!'); </script>";
	header("refresh: 1; url=../create_ac.html");
}
else
{
	if ($conn->query($sql) == TRUE)
	{
		header("refresh: 1; url = ../index.php");	
		echo "<script> alert('Success') </script>";
	}     		
	else 
	{
    	echo "Error: " . $sql . "<br>" . $conn->error;
	}
}

$conn->close();

?>